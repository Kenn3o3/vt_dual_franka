# VT_Franka remote training

Remote host layout:

```text
/mnt/pfs_cuhk/kenny/vt_franka
```

Workflow:

1. Sync repo code.
2. Sync `robot_workspace/data/preprocess1/<task>/<profile>/`.
3. Run each model in its own `tmux` session on the remote host.
4. Upload/download best checkpoints with ModelScope, or rsync checkpoints back to local.

This remote path only trains. It does not collect raw episodes.
