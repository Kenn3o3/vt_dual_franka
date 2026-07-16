#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-$(dirname "$0")/../config/workspace.yaml}"
vt-dual-franka-workspace gelsight --workspace-config "$CONFIG_PATH"
