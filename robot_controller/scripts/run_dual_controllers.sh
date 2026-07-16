#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTROLLER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${CONTROLLER_DIR}/.." && pwd)"

python "${SCRIPT_DIR}/preflight_dual_network.py" --skip-ping

export PYTHONPATH="${REPO_ROOT}/shared/src:${CONTROLLER_DIR}/src:${PYTHONPATH:-}"

LEFT_CONTROLLER_CPUS="${LEFT_CONTROLLER_CPUS:-4}"
RIGHT_CONTROLLER_CPUS="${RIGHT_CONTROLLER_CPUS:-3,7}"
CONTROLLER_START_DELAY_SEC="${CONTROLLER_START_DELAY_SEC:-10}"

taskset -c "${LEFT_CONTROLLER_CPUS}" \
  python -m vt_dual_franka_controller.cli run --config "${CONTROLLER_DIR}/config/controller_left.yaml" &
LEFT_PID="$!"
sleep "${CONTROLLER_START_DELAY_SEC}"
if ! kill -0 "${LEFT_PID}" 2>/dev/null; then
  wait "${LEFT_PID}"
fi
taskset -c "${RIGHT_CONTROLLER_CPUS}" \
  python -m vt_dual_franka_controller.cli run --config "${CONTROLLER_DIR}/config/controller_right.yaml" &
RIGHT_PID="$!"

trap 'kill "${LEFT_PID}" "${RIGHT_PID}" 2>/dev/null || true' INT TERM EXIT
wait -n "${LEFT_PID}" "${RIGHT_PID}"
