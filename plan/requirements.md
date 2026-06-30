# Clean Visuotactile Pipeline Requirements

## Terms

- `camera standardization`: live camera-level conversion shared by collection and inference.
- `preprocess1`: synchronized model-agnostic dataset, not a collection-time image resize recorder.
- `preprocess2`: model-specific image/state preprocessing owned by a policy adapter and locked into checkpoint manifests.
- `common dataset`: self-contained image-file plus index dataset produced by `make-dataset`.

## Confirmed Requirements

- Standard camera frames are RGB `uint8` with shape `(480, 640, 3)`.
- Wrist RGB stream name is `rgb_wrist`.
- The single GelSight Mini stream name is `tactile_left`.
- Collection stores lightweight standardized images, not native 2xxx x 3xxx GelSight frames.
- Collection should avoid per-frame JPEG write/encode in camera threads during active recording.
- Collection buffers standardized frames in memory and flushes JPEG q90 plus indexes at episode stop.
- `make-dataset` points to a task collect folder and generates a self-contained common dataset.
- Common dataset generation uses fixed 10Hz causal alignment.
- Training commands consume the common dataset.
- Inference camera workers output standardized frames, then policy code applies only preprocess2.
- V1 inference uses sequential closed-loop chunk execution, not RDP async action-buffer execution.
- Old checkpoints do not need compatibility support.

## Defaults

- Standard resolution: `640x480`.
- Standard color: RGB.
- Collection encoding: JPEG q90.
- Common dataset Hz: 10.
- Hardware gate duration: 30 seconds.
- Hardware gate thresholds: GelSight/tactile_left >= 9.0Hz, wrist >= 20Hz.
- Observation settle in inference is configurable and default off.

## Non-Goals

- No fallback to collection-time `CanonicalPreprocess1StreamRecorder`.
- No runtime compatibility branch for old checkpoints.
- No fake left/right tactile duplication for new models.
- No RDP-style async action buffer in v1.
