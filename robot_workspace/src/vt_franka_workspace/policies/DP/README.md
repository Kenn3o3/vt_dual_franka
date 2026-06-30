# DP Policy Family

This directory is self-contained for the paper's diffusion-policy baselines:

- `manifeel`: wrist RGB + tactile marker RGB + proprioception with Manifeel-style fusion.
- `equidiff_tact`: wrist RGB + tactile marker RGB + proprioception with EquiDiff tactile fusion.

The Hydra configs, workspaces, datasets, diffusion policy code, and deployment
code live under `policy/DP/`. This family does not import or vendor the ISP
backend; ISP remains isolated under `policy/ISP`.

Train from the repo root:

```bash
python -m policy.DP.train <task> --experiment-name <exp> --config-name <manifeel|equidiff_tact> --n-demo <N>
```

Evaluation uses:

```bash
python -m univtac.eval.runner --ckpt <checkpoint_path> --inference-config <cfg> --total-num <N>
```
