# ACT Policy Family

This directory is self-contained for the paper's ACT-family baselines:

- `univtac`: wrist RGB + tactile marker RGB + proprioception with the UniVTAC tactile encoder.

The ACT policy wrapper, dataset conversion code, deployment code, and original
ACT/DETR implementation all live under `policy/ACT/`. This family does not import
or vendor the ISP backend.

Train from the repo root:

```bash
python -m policy.ACT.train <task> --experiment-name <exp> --config-name univtac --n-demo <N>
```

Evaluation uses:

```bash
python -m univtac.eval.runner --ckpt <checkpoint_path> --inference-config <cfg> --total-num <N>
```
