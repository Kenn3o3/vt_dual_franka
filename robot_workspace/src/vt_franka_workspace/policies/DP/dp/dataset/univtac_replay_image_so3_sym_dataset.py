import numpy as np

from dp.common.normalize_util import (
    abs_action_only_symmetric_normalizer_from_stat,
    array_to_stats,
    get_identity_normalizer_from_stat,
    get_image_identity_normalizer,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
    get_range_symmetric_normalizer_from_stat,
)
from dp.dataset.univtac_replay_image_dataset import (
    UniVTACReplayImageDataset,
    normalizer_from_stat,
)
from dp.model.common.normalizer import LinearNormalizer


class UniVTACReplayImageSO3SymDataset(UniVTACReplayImageDataset):
    def __init__(
        self,
        shape_meta: dict,
        dataset_root: str,
        task_name: str,
        split: str = "clean",
        n_demo: int = 100,
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        n_obs_steps=None,
        abs_action: bool = True,
        use_legacy_normalizer: bool = False,
        normalization_mode: str = "default",
        use_cache: bool = True,
        seed: int = 42,
        val_ratio: float = 0.0,
        ws_center_source: str = "auto",
        ws_x_center: float = 0.0,
        ws_y_center: float = 0.0,
        ws_z_center: float = 0.8,
        cache_dir: str = None,
        **kwargs,
    ):
        super().__init__(
            shape_meta=shape_meta,
            dataset_root=dataset_root,
            task_name=task_name,
            split=split,
            n_demo=n_demo,
            horizon=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            n_obs_steps=n_obs_steps,
            abs_action=abs_action,
            use_legacy_normalizer=use_legacy_normalizer,
            normalization_mode=normalization_mode,
            use_cache=use_cache,
            seed=seed,
            val_ratio=val_ratio,
            image_identity_normalizer=False,
            cache_dir=cache_dir,
            **kwargs,
        )
        if ws_center_source == "auto":
            self.ws_center = self.ws_center.astype(np.float32)
        elif ws_center_source == "manual":
            self.ws_center = np.array(
                [ws_x_center, ws_y_center, ws_z_center], dtype=np.float32
            )
        else:
            raise ValueError(f"Unsupported ws_center_source: {ws_center_source}")

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        def _centered_stat(raw_stat):
            stat = {k: np.copy(v) for k, v in raw_stat.items()}
            for key in ("min", "max", "mean"):
                stat[key][:3] -= self.ws_center
            return stat

        stat = _centered_stat(array_to_stats(self.replay_buffer["action"]))
        if self.normalization_mode == "off":
            action_normalizer = get_identity_normalizer_from_stat(stat)
        elif self.abs_action:
            action_normalizer = abs_action_only_symmetric_normalizer_from_stat(
                stat
            )
            if self.use_legacy_normalizer:
                action_normalizer = normalizer_from_stat(stat)
        else:
            action_normalizer = get_identity_normalizer_from_stat(stat)
        normalizer["action"] = action_normalizer

        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])
            if key.endswith("pos") and not key.endswith("qpos"):
                stat = _centered_stat(stat)
            if self.normalization_mode == "off":
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith("qpos"):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith("pos"):
                this_normalizer = get_range_symmetric_normalizer_from_stat(stat)
            elif key.endswith("quat"):
                this_normalizer = get_identity_normalizer_from_stat(stat)
            else:
                raise RuntimeError(f"Unsupported lowdim key '{key}'")
            normalizer[key] = this_normalizer

        for key in self.rgb_keys:
            if self.normalization_mode == "off":
                normalizer[key] = get_image_identity_normalizer()
            else:
                normalizer[key] = get_image_range_normalizer()

        normalizer["pos_vecs"] = get_identity_normalizer_from_stat(
            {
                "min": -1 * np.ones([10, 2], np.float32),
                "max": np.ones([10, 2], np.float32),
            }
        )
        if self.normalization_mode == "off":
            normalizer["crops"] = get_image_identity_normalizer()
        else:
            normalizer["crops"] = get_image_range_normalizer()
        return normalizer
