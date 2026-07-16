# VT Dual Franka

## Layout

- `shared/`: cross-machine models, math, calibration, timing, and interpolation utilities.
- `robot_controller/`: Polymetis-backed high-frequency controller for the Franka-side machine.
- `robot_workspace/`: Quest teleop, Quest feedback publishing, GelSight and Orbbec handling, recording, and rollout utilities.
- `docs/`: setup, architecture, and migration notes.
- `ops/`: shell helpers for common launch patterns.
