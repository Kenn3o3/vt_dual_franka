# Hardware Acceptance

## Command

```bash
vt-franka-workspace diagnose-cameras \
  --workspace-config robot_workspace/config/workspace.yaml \
  --task usb_insertion \
  --duration-sec 30
```

## Default Gates

- `tactile_left` GelSight standardized capture: at least 9.0Hz.
- `rgb_wrist` standardized capture: at least 20Hz.
- Dual-camera simultaneous capture keeps GelSight at least 9.0Hz.

## Stages

1. GelSight-only standardization throughput.
2. Wrist-only standardization throughput.
3. Simultaneous GelSight + wrist memory buffering.
4. JPEG q90 episode flush latency.
5. Inference observation loop using fake policy/no robot motion.

## Report

Write JSON to `analysis/camera_diagnostics/<timestamp>/report.json`.

Include:

- effective Hz
- p50/p95/max inter-frame gap
- resize latency p50/p95
- stale/dropped count
- memory estimate
- JPEG flush latency
- pass/fail and failed checks
