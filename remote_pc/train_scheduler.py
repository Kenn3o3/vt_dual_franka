from __future__ import annotations

import argparse
import json
import queue
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


DEFAULT_REMOTE_ROOT = Path("/mnt/pfs_cuhk/kenny/vt_franka")
DEFAULT_PLAN = DEFAULT_REMOTE_ROOT / "remote_pc" / "plans" / "usb_insertion_all.txt"
DEFAULT_STATE_DIR = DEFAULT_REMOTE_ROOT / "robot_workspace" / "data" / "checkpoints" / "_remote_scheduler"
DEFAULT_PROFILE = "real_canonical_v1"

MODEL_MEMORY_GB = {
    "dp_manifeel": 24,
    "dp_equidiff_tact": 24,
    "act_univtac": 28,
    "vital_act": 24,
    "vital_dp": 24,
    "vista_so2": 36,
    "vista_so3": 36,
}


@dataclass(frozen=True)
class ScheduledJob:
    task_name: str
    model: str
    seed: int
    batch_size: int
    epochs: int
    run_name: str
    status: str = "pending"
    gpu: int | None = None
    checkpoint_dir: str | None = None
    command: list[str] | None = None
    started_at: float | None = None
    finished_at: float | None = None
    exit_code: int | None = None


def main() -> None:
    args = build_arg_parser().parse_args()
    remote_root = Path(args.remote_root)
    state_dir = Path(args.state_dir or (remote_root / "robot_workspace/data/checkpoints/_remote_scheduler"))
    jobs = load_plan(args.plan)
    jobs = expand_jobs(jobs, seeds=args.seeds, batch_size=args.batch_size, epochs=args.epochs)
    state_path = state_dir / "state.json"

    if args.status_only:
        print(state_path.read_text(encoding="utf-8") if state_path.exists() else json.dumps({"jobs": []}, indent=2))
        return

    if args.dry_run:
        payload = {
            "remote_root": str(remote_root),
            "state_dir": str(state_dir),
            "parallel": bool(args.parallel),
            "gpus": args.gpus,
            "jobs": [asdict(job) for job in jobs],
        }
        print(json.dumps(payload, indent=2))
        return

    state_dir.mkdir(parents=True, exist_ok=True)

    state = load_state(state_path)
    state["remote_root"] = str(remote_root)
    state["generated_at"] = time.time()
    state["jobs"] = [asdict(job) for job in jobs]
    write_state(state_path, state)

    if args.parallel:
        failures = run_jobs_parallel(
            jobs,
            remote_root=remote_root,
            state_path=state_path,
            state_dir=state_dir,
            args=args,
        )
        if failures:
            raise SystemExit(1)
    else:
        for job in jobs:
            run_job(job, remote_root=remote_root, state_path=state_path, state_dir=state_dir, args=args)


def run_jobs_parallel(
    jobs: list[ScheduledJob],
    *,
    remote_root: Path,
    state_path: Path,
    state_dir: Path,
    args: argparse.Namespace,
) -> list[ScheduledJob]:
    work_queue: queue.Queue[ScheduledJob] = queue.Queue()
    for job in jobs:
        work_queue.put(job)

    state_lock = threading.Lock()
    failures: list[ScheduledJob] = []
    failure_lock = threading.Lock()

    def worker(gpu: int) -> None:
        while True:
            try:
                job = work_queue.get_nowait()
            except queue.Empty:
                return
            try:
                exit_code = run_job(
                    job,
                    remote_root=remote_root,
                    state_path=state_path,
                    state_dir=state_dir,
                    args=args,
                    fixed_gpu=gpu,
                    raise_on_failure=False,
                    state_lock=state_lock,
                )
                if exit_code != 0:
                    with failure_lock:
                        failures.append(job)
            except Exception as exc:
                failed_job = ScheduledJob(
                    **{
                        **asdict(job),
                        "status": "failed",
                        "gpu": gpu,
                        "started_at": time.time(),
                        "finished_at": time.time(),
                        "exit_code": -1,
                    }
                )
                _update_job_state(state_path, failed_job, state_lock=state_lock)
                log_path = state_dir / "logs" / f"{job.run_name}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("ab") as log_file:
                    log_file.write(f"\n[scheduler] {type(exc).__name__}: {exc}\n".encode("utf-8"))
                with failure_lock:
                    failures.append(failed_job)
            finally:
                work_queue.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in args.gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return failures


def run_job(
    job: ScheduledJob,
    *,
    remote_root: Path,
    state_path: Path,
    state_dir: Path,
    args: argparse.Namespace,
    fixed_gpu: int | None = None,
    raise_on_failure: bool = True,
    state_lock: threading.Lock | None = None,
) -> int:
    gpu = select_gpu(job.model, [fixed_gpu] if fixed_gpu is not None else args.gpus, args.gpu_buffer_gb)
    job = ScheduledJob(**{**asdict(job), "status": "running", "started_at": time.time(), "gpu": gpu})
    _update_job_state(state_path, job, state_lock=state_lock)
    remote_checkpoint_dir = remote_root / "robot_workspace" / "data" / "checkpoints" / job.task_name / "visuotactile" / job.model
    remote_prepared_dir = remote_root / "robot_workspace" / "data" / "prepared" / job.task_name / "visuotactile" / "real_canonical_v1" / job.model

    prepare_cmd = [
        sys.executable,
        "-m",
        "vt_franka_workspace.policies.visuotactile.prepare",
        "--workspace-config",
        str(remote_root / "robot_workspace/config/workspace.yaml"),
        "--task-name",
        job.task_name,
        "--model",
        job.model,
        "--raw-run-dir",
        str(remote_root / "robot_workspace/data/collect" / job.task_name),
        "--output-dir",
        str(remote_prepared_dir),
        "--source",
        "preprocess1",
        "--source-root",
        str(remote_root / "robot_workspace/data/preprocess1" / job.task_name / DEFAULT_PROFILE),
        "--overwrite",
    ]
    train_cmd = [
        sys.executable,
        "-m",
        "vt_franka_workspace.policies.visuotactile.train",
        "--workspace-config",
        str(remote_root / "robot_workspace/config/workspace.yaml"),
        "--task-name",
        job.task_name,
        "--model",
        job.model,
        "--dataset-dir",
        str(remote_prepared_dir),
        "--checkpoint-dir",
        str(remote_checkpoint_dir),
        "--backend-dataset-root",
        str(remote_checkpoint_dir / "backend_dataset"),
        "--seed",
        str(job.seed),
        "--batch-size",
        str(job.batch_size),
        "--epochs",
        str(job.epochs),
        "--wandb-mode",
        "offline",
        "--device",
        "cuda:0",
        "--no-prepare",
        "--overwrite",
    ]
    cuda_visible_devices = "" if job.gpu is None else f"export CUDA_VISIBLE_DEVICES={job.gpu} && "
    pythonpath_prefix = f"{remote_root / 'robot_workspace/src'}:{remote_root / 'shared/src'}"
    shell_command = (
        f"cd {shlex.quote(str(remote_root))} && "
        f"export PYTHONPATH={shlex.quote(pythonpath_prefix)}:$PYTHONPATH && "
        f"{shlex.join(prepare_cmd)} && {cuda_visible_devices}{shlex.join(train_cmd)}"
    )
    command = ["bash", "-lc", shell_command]
    job = ScheduledJob(**{**asdict(job), "command": command, "checkpoint_dir": str(remote_checkpoint_dir)})
    _update_job_state(state_path, job, state_lock=state_lock)
    log_path = state_dir / "logs" / f"{job.run_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log_file:
        proc = subprocess.Popen(command, cwd=str(remote_root), stdout=log_file, stderr=subprocess.STDOUT)
        exit_code = proc.wait()
    job = ScheduledJob(**{**asdict(job), "status": "finished" if exit_code == 0 else "failed", "finished_at": time.time(), "exit_code": exit_code})
    _update_job_state(state_path, job, state_lock=state_lock)
    if exit_code != 0 and raise_on_failure:
        raise SystemExit(exit_code)
    return exit_code


def load_plan(path: Path) -> list[dict[str, Any]]:
    lines = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip() and not line.strip().startswith("#")]
    if len(lines) % 2 != 0:
        raise ValueError("Plan file must contain task and model lines in pairs")
    jobs: list[dict[str, Any]] = []
    for i in range(0, len(lines), 2):
        task_name = lines[i]
        models = lines[i + 1].split()
        for model in models:
            jobs.append({"task_name": task_name, "model": model})
    return jobs


def expand_jobs(jobs: list[dict[str, Any]], *, seeds: list[int], batch_size: int, epochs: int) -> list[ScheduledJob]:
    expanded: list[ScheduledJob] = []
    for job in jobs:
        task_name = str(job["task_name"])
        model = str(job["model"])
        for seed in seeds:
            run_name = f"{task_name}_{model}"
            expanded.append(
                ScheduledJob(
                    task_name=task_name,
                    model=model,
                    seed=int(seed),
                    batch_size=int(batch_size),
                    epochs=int(epochs),
                    run_name=run_name,
                )
            )
    return expanded


def select_gpu(model: str, gpus: list[int], gpu_buffer_gb: float) -> int:
    required_mb = int((MODEL_MEMORY_GB.get(model, 24) + float(gpu_buffer_gb)) * 1024)
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception:
        return gpus[0]
    free_by_gpu: dict[int, int] = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        index_text, free_text = [item.strip() for item in line.split(",", 1)]
        free_by_gpu[int(index_text)] = int(free_text)
    candidates = [(gpu, free_by_gpu.get(gpu, 0)) for gpu in gpus]
    candidates.sort(key=lambda item: item[1], reverse=True)
    if not candidates:
        raise RuntimeError("No GPUs configured for scheduler")
    best_gpu, best_free = candidates[0]
    if best_free < required_mb:
        raise RuntimeError(f"No GPU has enough free memory for {model}: need {required_mb} MiB, best gpu {best_gpu} has {best_free} MiB")
    return best_gpu


def load_state(state_path: Path) -> dict[str, Any]:
    if state_path.exists():
        return json.loads(state_path.read_text(encoding="utf-8"))
    return {"jobs": []}


def write_state(state_path: Path, payload: dict[str, Any]) -> None:
    state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def update_job_state(state_path: Path, job: ScheduledJob) -> None:
    state = load_state(state_path)
    jobs = state.setdefault("jobs", [])
    updated = False
    for index, payload in enumerate(jobs):
        if payload.get("run_name") == job.run_name:
            jobs[index] = asdict(job)
            updated = True
            break
    if not updated:
        jobs.append(asdict(job))
    write_state(state_path, state)


def _update_job_state(state_path: Path, job: ScheduledJob, *, state_lock: threading.Lock | None = None) -> None:
    if state_lock is None:
        update_job_state(state_path, job)
        return
    with state_lock:
        update_job_state(state_path, job)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VT_Franka remote visuotactile training scheduler")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--remote-root", type=Path, default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--state-dir", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--gpus", type=_parse_gpus, default=[0])
    parser.add_argument("--gpu-buffer-gb", type=float, default=8.0)
    parser.add_argument("--parallel", action="store_true", help="Run one worker per configured GPU.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    return parser


def _parse_gpus(value: str) -> list[int]:
    gpus = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not gpus:
        raise argparse.ArgumentTypeError("--gpus must contain at least one GPU index")
    return gpus


if __name__ == "__main__":
    main()
