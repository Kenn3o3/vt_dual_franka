# Data Contracts

## Standardized Camera Frame

- `image`: numpy array, RGB, `uint8`, shape `(480, 640, 3)`.
- `metadata.stream_name`: canonical stream name.
- `metadata.camera_name`: physical camera name.
- `metadata.captured_wall_time`: Python wall time.
- `metadata.sequence_id`: monotonically increasing per camera when available.
- `metadata.source_shape`: original frame shape.
- `metadata.standard_shape`: `[480, 640, 3]`.
- `metadata.color_space`: `RGB`.
- `metadata.standardization`: name/version of camera standardization.

## Collection Episode Streams

- `streams/rgb_wrist/frames/000000.jpg`
- `streams/rgb_wrist/index.jsonl`
- `streams/rgb_wrist/manifest.json`
- `streams/tactile_left/frames/000000.jpg`
- `streams/tactile_left/index.jsonl`
- `streams/tactile_left/manifest.json`

Each image index record contains:

- `frame_index`
- `frame_path`
- `captured_wall_time`
- `sequence_id`
- `frame_width`
- `frame_height`
- `metadata`

## Common Dataset

Root: `robot_workspace/data/datasets/<task>/<dataset_name>`.

- `dataset_manifest.json`
- `episodes/<episode_id>/steps.jsonl`
- `episodes/<episode_id>/images/rgb_wrist/*.jpg`
- `episodes/<episode_id>/images/tactile_left/*.jpg`
- optional low-dimensional arrays/manifests

Each step record contains:

- `episode_id`
- `step_index`
- `timestamp`
- image paths for `rgb_wrist` and `tactile_left`
- selected source frame metadata
- controller state
- action/teleop command when available

## Checkpoint Bundle

- `policy_manifest.json`
- `dataset_manifest.json` or dataset hash/reference
- `preprocess2_manifest.json`
- `normalizer_stats.json`
- model checkpoint artifact(s)
