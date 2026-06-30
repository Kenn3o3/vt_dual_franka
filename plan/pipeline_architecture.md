# Pipeline Architecture

## Collection

1. Device-specific camera reader obtains a raw frame.
2. Shared camera standardizer converts it to RGB `uint8` `(480, 640, 3)`.
3. Standardized frame updates the live buffer for preview/inference-style observation.
4. During active episode recording, standardized frame is appended to an in-memory episode stream buffer.
5. On episode stop, stream buffers flush JPEG q90 files plus `index.jsonl` and per-stream manifests.

## Dataset Generation

1. `make-dataset <task_collect_folder>` discovers complete saved episodes.
2. It reads stream indexes for `rgb_wrist`, `tactile_left`, controller state, and teleop commands.
3. It builds a fixed 10Hz grid.
4. For each grid timestamp, it selects latest observations at or before the grid time and the next action after the grid time.
5. It copies selected JPEGs into a self-contained dataset and writes `steps.jsonl` and `dataset_manifest.json`.

## Training

1. `train-policy --model <model> --dataset <common_dataset>` reads the common dataset.
2. The model adapter resolves its preprocess2 defaults.
3. Prepared cache is built or reused for that model.
4. Checkpoint bundle records policy, dataset, preprocess2, and normalizer manifests.

## Inference

1. `run-policy --workspace-config ... --task <task> --policy-config <policy>` loads task camera config.
2. Camera workers populate standardized live buffers.
3. Observation assembler validates frame age and standardized frame contract.
4. Policy adapter reads manifest-locked preprocess2 and converts observation windows to tensors.
5. Runner executes predicted action chunks sequentially and refreshes the observation window after the final `obs_horizon` actions.

## Module Boundaries

- Camera modules know device APIs and camera standardization only.
- Recording modules know episode stream buffers and image/index persistence.
- Dataset modules know synchronization and self-contained dataset layout.
- Policy modules know preprocess2 and model tensor contracts.
- Inference runner knows control-loop sequencing but not image resizing details.
