import hydra
import numpy as np
import math
import psutil
import torch
import cv2
import swanlab
import time

from omegaconf import DictConfig
from tqdm import tqdm
from typing import Dict, List, Optional
from pathlib import Path

from movement_primitive_diffusion.agents.base_agent import BaseAgent
from movement_primitive_diffusion.utils.gym_utils.async_vector_env import AsyncVectorEnv
from movement_primitive_diffusion.utils.gym_utils.sync_vector_env import SyncVectorEnv
from movement_primitive_diffusion.utils.helper import list_of_dicts_of_arrays_to_dict_of_arrays
from movement_primitive_diffusion.utils.video import save_video_from_array, save_gif_from_array
from movement_primitive_diffusion.utils.visualization import tile_images


class BaseVectorWorkspace:
    def __init__(
        self,
        env_config: DictConfig,
        t_act: int,
        num_parallel_envs: int,
        shared_memory: bool = False,
        async_vector_env: bool = True,
        num_upload_successful_videos: int = 5,
        num_upload_failed_videos: int = 5,
        video_dt: float = 0.1,
        show_images: bool = False,
        annotate_videos: bool = False,
        timeout: Optional[float] = None,
    ):
        super().__init__()

        self.env_config = env_config
        self.num_parallel_envs = num_parallel_envs
        self.shared_memory = shared_memory
        self.async_vector_env = async_vector_env
        self.timeout = timeout
        # Defer env creation to first test_agent() call to avoid occupying CPU
        # with idle SOFA processes during pure training epochs.
        self.vector_env = None

        self.t_act = t_act
        self.time_limit = env_config.time_limit
        self.fps = 1 / video_dt
        self.show_images = show_images

        self.num_upload_successful_videos = num_upload_successful_videos
        self.num_upload_failed_videos = num_upload_failed_videos

        self.annotate_videos = annotate_videos

    def create_vectorized_env(self):
        # Create vectorized environment
        self.envs_fns = [(lambda i=i: hydra.utils.instantiate(self.env_config, seed=i)) for i in range(self.num_parallel_envs)]

        if self.async_vector_env:
            # spawn: each worker starts a fresh Python interpreter with no
            # inherited OpenGL/EGL/CUDA state.  LD_PRELOAD is re-applied by
            # the dynamic linker, so fix_sofa_headless.so intercepts GL calls
            # before any library static initialisers run.
            # With spawn, Sofa.SofaGL is imported only after _init_pyglet_window()
            # creates an active EGL context, so GLEW initialisation succeeds.
            self.vector_env = AsyncVectorEnv(self.envs_fns, daemon=False,
                                             context='spawn',
                                             shared_memory=self.shared_memory)
            # Give each subprocess a group of cores rather than a single core.
            # Mesa llvmpipe is multi-threaded; pinning to 1 core serializes its
            # tile-rendering threads and causes all envs to appear sequential.
            try:
                available_cpus = psutil.Process().cpu_affinity()
                available_cpu_count = len(available_cpus)
                # Each env gets at least 2 cores; distribute evenly.
                cores_per_env = max(2, available_cpu_count // self.num_parallel_envs)
                env_pids = [self.vector_env.processes[i].pid
                            for i in range(self.num_parallel_envs)]
                for i, pid in enumerate(env_pids):
                    p = psutil.Process(pid)
                    start = (i * cores_per_env) % available_cpu_count
                    assigned = [available_cpus[(start + c) % available_cpu_count]
                                for c in range(cores_per_env)]
                    p.cpu_affinity(cpus=assigned)
            except Exception:
                pass  # CPU affinity is best-effort
        else:
            self.vector_env = SyncVectorEnv(self.envs_fns)

    def _destroy_vectorized_env(self) -> None:
        """Shut down all SOFA subprocesses and release CPU/memory resources."""
        if self.vector_env is not None:
            try:
                self.vector_env.close()
            except Exception:
                pass
            self.vector_env = None

    def test_agent(self, agent: BaseAgent, num_trajectories: int = 10) -> dict:
        # Create SOFA subprocesses right before eval starts
        if self.vector_env is None:
            self.create_vectorized_env()

        # How many times we run trajectories in the vectorized environment
        self.num_sequential_episodes = math.ceil(num_trajectories / self.num_parallel_envs)

        # How ofter we call agent.predict in total
        self.max_action_sequences = math.ceil(self.time_limit / self.t_act)

        # How many steps we run in total
        self.global_max_action_sequences = self.num_sequential_episodes * self.max_action_sequences
        self.global_max_steps = self.global_max_action_sequences * self.t_act

        # Create progress bar
        self.progress_bar = tqdm(range(self.global_max_steps), desc="Testing agent", leave=False)
        global_steps_to_go = self.global_max_steps

        # Variables for video logging
        self.num_successful_trajectories = 0
        self.num_failed_trajectories = 0
        frames_of_successful_trajectories = []
        frames_of_failed_trajectories = []
        
        # Store individual episode videos
        self.episode_videos = []

        # Execute num_sequential_episodes episodes in the vectorized environment
        for episode_sequence_index in range(self.num_sequential_episodes):
            self.reset_env(caller_locals=locals())

            # List for images that are shown to the user
            image_tuple = self.render_function(caller_locals=locals())
            image_shape = image_tuple[0].shape
            if self.show_images:
                display_images = np.zeros((self.num_parallel_envs, *image_shape), dtype=np.uint8)
                green_image = np.zeros_like(image_tuple[0])
                green_image[:, :, 1] = 255
                red_image = np.zeros_like(image_tuple[0])
                red_image[:, :, 0] = 255

            # Variables to keep track of the env's done and successful status, and rendered frames
            done_buffer = [False] * self.num_parallel_envs
            successful_buffer = [False] * self.num_parallel_envs
            frame_buffer = [[] for _ in range(self.num_parallel_envs)]

            # Execute max_action_sequences action sequences in the vectorized environment
            for action_sequence_index in range(self.max_action_sequences):
                # get_observation_dict returns a tuple of dicts of arrays -> move the length of the tuple into the batch dimension of the arrays
                observation_buffer_tuple = self.vector_env.call("get_observation_dict")
                observation_buffer = list_of_dicts_of_arrays_to_dict_of_arrays(observation_buffer_tuple)

                # Create torch tensors from numpy arrays
                for key, val in observation_buffer.items():
                    observation_buffer[key] = torch.from_numpy(val)

                # Process the observation buffer to get observations and extra inputs
                observation, extra_inputs = agent.process_batch.process_env_observation(observation_buffer)

                # Move observations and extra inputs to device
                for key, val in observation.items():
                    observation[key] = val.to(agent.device)
                for key, val in extra_inputs.items():
                    if isinstance(val, torch.Tensor):
                        extra_inputs[key] = val.to(agent.device)

                # Predict the next action sequence
                actions = agent.predict(observation, extra_inputs)

                # predict() already returns (B, T, N); move to cpu and numpy
                actions = actions.cpu().numpy()
                assert actions.ndim == 3 and actions.shape[0] == self.num_parallel_envs, f"Actions should be of shape (B, T, N) for B parallel environments, T timesteps, and N action dimensions. Got shape {actions.shape}"

                # Execute at most t_act actions of the sequence in the vectorized environment
                for action_step_index in range(self.t_act):
                    # Execute the actions in the vectorized environment and get the rendered frames
                    env_obs, env_reward, env_terminated, env_truncated, env_info = self.vector_env.step(actions[:, action_step_index, :], timeout=self.timeout)

                    post_step_frames = self.render_function(caller_locals=locals())

                    # Update the progress bar
                    global_steps_to_go -= 1
                    self.progress_bar.n = self.global_max_steps - global_steps_to_go
                    self.progress_bar.refresh()

                    # Update buffer for done and successful environments, and add frames to the frame buffer
                    global_step = self.global_max_steps - global_steps_to_go
                    for env_index in range(self.num_parallel_envs):
                        if not done_buffer[env_index]:
                            trajectory_index = episode_sequence_index * self.num_parallel_envs + env_index
                            # Only add frames to the frame buffer if the environment is not done and belongs to the num_trajectories
                            if trajectory_index < num_trajectories:
                                frame = post_step_frames[env_index].copy()
                                # Stamp step number so consecutive identical frames are distinguishable
                                # (prevents GIF encoders from collapsing all frames into one)
                                cv2.putText(frame, f"s{global_step}", (5, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
                                frame_buffer[env_index].append(frame)

                            # Update the display image
                            if self.show_images and trajectory_index < num_trajectories:
                                display_images[env_index] = post_step_frames[env_index]

                            # Check if the environment is done and if it was successful
                            if env_truncated[env_index] or env_terminated[env_index]:
                                done_buffer[env_index] = True
                                successful_buffer[env_index] = self.check_success_hook(locals())

                                # Overlay a green or red image on top of the frame to indicate success or failure
                                if self.show_images and trajectory_index < num_trajectories:
                                    if successful_buffer[env_index]:
                                        display_images[env_index] = cv2.addWeighted(display_images[env_index], 0.5, green_image, 0.5, 0)
                                    else:
                                        display_images[env_index] = cv2.addWeighted(display_images[env_index], 0.5, red_image, 0.5, 0)

                    # Call the post step hook
                    self.post_step_hook(caller_locals=locals())

                    # Display the images as a tiled grid
                    if self.show_images:
                        cv2.imshow("VectorWorkspace", tile_images(display_images)[..., ::-1])
                        cv2.waitKey(1)  # wait for 1 ms

                    # If all environments are done while executing the steps of an action sequence, do not execute the remaining steps
                    if all(done_buffer):
                        global_steps_to_go -= self.t_act - action_step_index - 1
                        break

                # If all environments are done before max_action_sequences are executed, do not execute the remaining action sequences
                if all(done_buffer):
                    global_steps_to_go -= (self.max_action_sequences - action_sequence_index - 1) * self.t_act
                    break

            # Before we start the next episode sequence, add the frames of the successful and failed trajectories to the video buffers
            for env_index in range(self.num_parallel_envs):
                trajectory_index = episode_sequence_index * self.num_parallel_envs + env_index
                if trajectory_index < num_trajectories:
                    env_frames = frame_buffer[env_index]
                    status = "success" if successful_buffer[env_index] else "failed"
                    
                    # Store individual episode video
                    self.episode_videos.append({
                        'episode_id': trajectory_index,
                        'frames': env_frames,
                        'status': status
                    })
                    
                    if successful_buffer[env_index]:
                        if self.num_successful_trajectories < self.num_upload_successful_videos:
                            if self.annotate_videos:
                                # Add a text overlay to the frame's top right corner with the trajectory number.
                                for frame in env_frames:
                                    cv2.putText(frame, f"{trajectory_index}", (frame.shape[1] - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            frames_of_successful_trajectories.extend(env_frames)
                        self.num_successful_trajectories += 1
                    else:
                        if self.num_failed_trajectories < self.num_upload_failed_videos:
                            if self.annotate_videos:
                                # Add a text overlay to the frames with the trajectory number.
                                for frame in env_frames:
                                    cv2.putText(frame, f"{trajectory_index}", (frame.shape[1] - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            frames_of_failed_trajectories.extend(env_frames)
                        self.num_failed_trajectories += 1

            # Display the current success rate in the progress bar
            self.progress_bar.set_postfix(success_rate=self.num_successful_trajectories / (self.num_successful_trajectories + self.num_failed_trajectories))

            # Call the post episode hook
            self.post_episode_hook(caller_locals=locals())

        # Log at least one black frame to have info for this epoch
        if len(frames_of_successful_trajectories) == 0:
            frames_of_successful_trajectories.append(np.zeros(image_shape, dtype=np.uint8))
        if len(frames_of_failed_trajectories) == 0:
            frames_of_failed_trajectories.append(np.zeros(image_shape, dtype=np.uint8))
        self.log_video(frames_of_successful_trajectories, fps=self.fps, metric="successful")
        self.log_video(frames_of_failed_trajectories, fps=self.fps, metric="failed")

        # Save individual episode videos
        self._save_individual_episode_videos()

        result = self.get_result_dict(caller_locals=locals())

        return result

    def check_success_hook(self, caller_locals: Dict) -> bool:
        """Function to modify success check behavior in subclasses.

        For example for checking if the agent reached the goal.

        """
        return caller_locals["env_terminated"][caller_locals["env_index"]]

    def reset_env(self, caller_locals: Dict) -> np.ndarray:
        """Function to modify reset behavior in subclasses.

        For example for setting a random seed, or passing an options dict.

        """
        return self.vector_env.reset()

    def render_function(self, caller_locals: Dict) -> np.ndarray:
        return self.vector_env.call("_render_frame", mode="rgb_array")

    def post_step_hook(self, caller_locals: Dict) -> None:
        """Function to modify post step behavior in subclasses.

        Updating the current best value of some metric.
        """
        pass

    def post_episode_hook(self, caller_locals: Dict) -> None:
        """Function to modify post episode behavior in subclasses.

        For example adding information to the progress bar.
        """
        pass

    def get_result_dict(self, caller_locals: Dict) -> Dict[str, float]:
        """Function to modify result dict in subclasses.

        For example adding information to the progress bar.
        """

        return {"success_rate": self.num_successful_trajectories / (self.num_successful_trajectories + self.num_failed_trajectories)}

    def get_result_dict_keys(self) -> List[str]:
        return ["success_rate"]

    def log_video(self, frame_buffer: List[np.ndarray], fps: int, metric: str = "video") -> None:
        # Write video - save to Hydra output directory
        now = time.strftime("%Y%m%d-%H%M%S")
        
        # Use Hydra output directory directly (respects method_name parameter)
        base_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        
        # Get current epoch (default to 0 if not set)
        current_epoch = getattr(self, 'current_epoch', 0)
        
        # Create epoch-specific directory
        epoch_dir = base_dir / f"epoch_{current_epoch:04d}"
        video_dir = epoch_dir / "media" / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)

        if swanlab.get_run() is not None:
            # Save as GIF for SwanLab visualization
            file_path = str(((video_dir / f"{metric}_{now}.gif").absolute()))
            save_gif_from_array(frames=frame_buffer, fps=fps, file_path=file_path, resize_height=None)
            swanlab.log({metric: swanlab.Video(file_path)})
        else:
            # Save as MP4 when SwanLab is disabled
            file_path = str(((video_dir / f"{metric}_{now}.mp4").absolute()))
            save_video_from_array(frames=frame_buffer, fps=fps, file_path=file_path, resize_height=None)

    def _save_individual_episode_videos(self):
        """Save individual episode videos with unified naming"""
        try:
            # Use Hydra output directory directly (respects method_name parameter)
            base_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
            
            # Get current epoch (default to 0 if not set)
            current_epoch = getattr(self, 'current_epoch', 0)
            
            # Create epoch-specific directory
            epoch_dir = base_dir / f"epoch_{current_epoch:04d}"
            episodes_dir = epoch_dir / "episodes"
            episodes_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\n🎬 Saving individual episode videos...")
            
            for episode_data in self.episode_videos:
                episode_id = episode_data['episode_id']
                frames = episode_data['frames']
                status = episode_data.get('status', 'unknown')

                if len(frames) > 0:
                    # Save as GIF (low-res, for quick preview)
                    gif_path = str((episodes_dir / f"{status}_episode_{episode_id:04d}.gif").absolute())
                    save_gif_from_array(frames=frames, fps=self.fps, file_path=gif_path, resize_height=None)

                    # Save the last frame as a lossless PNG (no color quantization artifacts)
                    last_frame_bgr = cv2.cvtColor(frames[-1], cv2.COLOR_RGB2BGR)
                    png_path = str((episodes_dir / f"{status}_episode_{episode_id:04d}_last_frame.png").absolute())
                    cv2.imwrite(png_path, last_frame_bgr)

            print(f"Saved {len(self.episode_videos)} episode videos + last-frame PNGs to epoch_{current_epoch:04d}")
            
        except Exception as e:
            print(f"⚠️  Warning: Failed to save episode videos: {e}")
    
    def close(self, timeout: Optional[float] = None) -> None:
        if self.vector_env is not None:
            self.vector_env.close(timeout=timeout)
