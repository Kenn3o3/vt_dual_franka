# Implementation Sequence

1. Add plan memory docs.
2. Add shared camera standardization and RGB JPEG IO helpers.
3. Add in-memory standardized episode stream buffers.
4. Wire collection camera workers to update live buffers and episode buffers.
5. Remove collection-time preprocess1 recorder usage from collection startup/stop/status.
6. Add `make-dataset` common dataset builder and CLI.
7. Update visuotactile runtime to consume standardized RGB frames and manifest-locked preprocess2.
8. Update inference runner to use task config and save debug input artifacts.
9. Add `diagnose-cameras` CLI.
10. Run focused unit tests and hardware diagnostics.

## Verification Order

1. `python -m pytest robot_workspace/tests/test_camera_standardization.py`
2. `python -m pytest robot_workspace/tests/test_episode_streams.py`
3. `python -m pytest robot_workspace/tests/test_common_dataset.py`
4. `python -m pytest robot_workspace/tests/test_visuotactile.py robot_workspace/tests/test_policy_runner.py`
5. `vt-franka-workspace diagnose-cameras ...`
