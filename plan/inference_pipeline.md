# Inference Pipeline

## Command Contract

Use task config for cameras and task-level runtime settings:

```bash
vt-franka-workspace run-policy \
  --workspace-config robot_workspace/config/workspace.yaml \
  --task usb_insertion \
  --policy-config robot_workspace/config/policies/visuotactile_usb_insertion_vista_so3.yaml
```

## Sequential Closed Loop

V1 keeps the understandable sequential loop:

1. Assemble observation window.
2. Call `policy.predict(observation_window)`.
3. Execute up to `exe_horizon` actions.
4. Collect observations after the final `obs_horizon` actions.
5. Use those observations for the next predict call.

Example: `obs_horizon=2`, `exe_horizon=8` collects observations after action 7 and action 8.

## Observation Contract

- `images.wrist.image` is standardized `rgb_wrist`.
- `tactile.tactile_left.image` is standardized GelSight.
- Assembler uses latest frame plus max-age checks.
- Policy runtime validates standardized shape, dtype, and RGB metadata.

## Preprocess2

- Model adapter code owns default preprocess2.
- Training writes resolved preprocess2 to checkpoint manifest.
- Inference reads manifest and calls the same adapter transform path.
- Default image transform is center crop `640x480 -> 480x480`, then resize to model size.

## Debug Artifacts

Every predict call saves under `data/eval/<task>/<run>/debug_inputs/predict_xxxxxx`:

- standardized `640x480` RGB images for every observation in the window
- right-before-model preprocess2 visualization
- metadata JSON with frame timestamps, ages, shapes, preprocess2 spec, and checkpoint manifest hash
