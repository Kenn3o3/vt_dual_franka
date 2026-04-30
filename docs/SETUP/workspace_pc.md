# Workspace PC Setup

Install the workspace environment:

```bash
cd /home/zhenya/kenny/visuotact/vt_franka
conda env create -f robot_workspace/environment.yml
conda activate vt-franka-workspace
pip install -e shared
pip install -e robot_workspace
```

Install the Orbbec wheel only on machines that use Orbbec cameras:

```bash
pip install third_party/pyorbbecsdk2-2.0.18-cp310-cp310-linux_x86_64.whl
```

Check `robot_workspace/config/workspace.yaml`:

- `controller.host`: controller PC IP.
- `quest_feedback.quest_ip`: Quest headset IP.
- `recording.collect_root`, `prepared_root`, `train_root`, `eval_root`, `checkpoints_root`: fixed data roots.
- `calibration.calibration_dir`: calibration JSON directory.

For MPD-family training and checkpoint inference:

```bash
conda activate mpd
cd /home/zhenya/kenny/visuotact/vt_franka
pip install -e shared
pip install -e robot_workspace
pip install -e robot_workspace/third_parties/mpd
```
