#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./syn_ckpts.sh
      Re-download current remote best.ckpt for every task/model.

  ./syn_ckpts.sh --resume
      Resume mode:
      - skip already valid local best.ckpt
      - re-download missing ones
      - delete and re-download incomplete/corrupted/stale ones

  ./syn_ckpts.sh -h | --help
EOF
}

RESUME=0
while (($# > 0)); do
  case "$1" in
    --resume)
      RESUME=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERR] Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

REMOTE_USER="zlkenny"
REMOTE_HOST="120.48.58.215"
REMOTE_PORT="538"
SSH_KEY="$HOME/.ssh/zlkenny_yzy673_ed25519"

REMOTE_BASE="/home/zlkenny/kenny/visuotact/UniVTAC/policy/DP/data/outputs/tasks"
LOCAL_BASE="/home/zhenya/kenny/visuotact/UniVTAC/policy/DP/data/outputs/tasks"

SSH_OPTS_SSH=(
  -p "$REMOTE_PORT"
  -i "$SSH_KEY"
  -o IdentitiesOnly=yes
  -o StrictHostKeyChecking=accept-new
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=6
)

SCP_OPTS=(
  -q
  -P "$REMOTE_PORT"
  -i "$SSH_KEY"
  -o IdentitiesOnly=yes
  -o StrictHostKeyChecking=accept-new
)

SSH_BASE=(ssh "${SSH_OPTS_SSH[@]}")
SCP_BASE=(scp "${SCP_OPTS[@]}")

mkdir -p -- "$LOCAL_BASE"

summary_tsv="${LOCAL_BASE}/sync_summary.tsv"
: > "$summary_tsv"
printf "task\tmodel\tkind\tlocal_path\tremote_path\trun\tepoch\tval_loss\tckpt_name\tsize_bytes\tstatus\n" >> "$summary_tsv"

# -----------------------------
# UI helpers
# -----------------------------
IS_TTY=0
[[ -t 1 ]] && IS_TTY=1

if (( IS_TTY )); then
  C_RESET=$'\033[0m'
  C_BOLD=$'\033[1m'
  C_DIM=$'\033[2m'
  C_RED=$'\033[31m'
  C_GREEN=$'\033[32m'
  C_YELLOW=$'\033[33m'
  C_BLUE=$'\033[34m'
  C_CYAN=$'\033[36m'
else
  C_RESET=""
  C_BOLD=""
  C_DIM=""
  C_RED=""
  C_GREEN=""
  C_YELLOW=""
  C_BLUE=""
  C_CYAN=""
fi

log() {
  printf "%b\n" "$*"
}

log_info() {
  log "${C_CYAN}[INFO]${C_RESET} $*"
}

log_ok() {
  log "${C_GREEN}[ OK ]${C_RESET} $*"
}

log_warn() {
  log "${C_YELLOW}[WARN]${C_RESET} $*"
}

log_err() {
  log "${C_RED}[ERR ]${C_RESET} $*"
}

human_bytes() {
  awk -v n="${1:-0}" 'BEGIN{
    split("B KB MB GB TB PB", u, " ");
    i=1;
    if (n < 0) n = 0;
    while (n >= 1024 && i < 6) { n /= 1024; i++ }
    if (i == 1) printf "%d %s", n, u[i];
    else printf "%.1f %s", n, u[i];
  }'
}

format_duration() {
  local s="${1:-0}"
  (( s < 0 )) && s=0
  printf "%02d:%02d:%02d" $((s/3600)) $(((s%3600)/60)) $((s%60))
}

estimate_finish_clock() {
  local eta="${1:-0}"
  local now
  now="$(date +%s)"
  if date -d "@$((now + eta))" '+%H:%M:%S' >/dev/null 2>&1; then
    date -d "@$((now + eta))" '+%H:%M:%S'
  else
    printf -- "--:--:--"
  fi
}

clear_line() {
  if (( IS_TTY )); then
    printf '\r\033[K'
  fi
}

render_progress() {
  local done="${1:-0}"
  local total="${2:-0}"
  local speed="${3:-0}"
  local item_idx="${4:-0}"
  local item_total="${5:-0}"
  local label="${6:-}"

  (( IS_TTY )) || return 0

  local pct=0
  local width=28
  local filled=0
  local remain=0
  local eta="--:--:--"
  local finish="--:--:--"

  if (( total > 0 )); then
    pct=$(( done * 100 / total ))
    (( pct > 100 )) && pct=100
    filled=$(( pct * width / 100 ))
    remain=$(( total - done ))
    (( remain < 0 )) && remain=0
  fi

  if (( speed > 0 && remain > 0 )); then
    local eta_sec
    eta_sec=$(( (remain + speed - 1) / speed ))
    eta="$(format_duration "$eta_sec")"
    finish="$(estimate_finish_clock "$eta_sec")"
  elif (( remain == 0 )); then
    eta="00:00:00"
    finish="$(estimate_finish_clock 0)"
  fi

  local bar rest
  printf -v bar '%*s' "$filled" ''
  printf -v rest '%*s' "$((width - filled))" ''
  bar=${bar// /█}
  rest=${rest// /░}

  printf '\r%s[%s%s]%s %3d%% | %s / %s | %s/s | Total ETA %s | Finish %s | %d/%d | %s' \
    "$C_BOLD" "$bar" "$rest" "$C_RESET" \
    "$pct" \
    "$(human_bytes "$done")" \
    "$(human_bytes "$total")" \
    "$(human_bytes "$speed")" \
    "$eta" \
    "$finish" \
    "$item_idx" \
    "$item_total" \
    "$label"
}

# -----------------------------
# Misc helpers
# -----------------------------
file_size() {
  stat -c %s -- "$1" 2>/dev/null || echo 0
}

calc_avg_speed() {
  if [[ -z "${TRANSFER_START_EPOCH:-}" ]]; then
    echo 0
    return 0
  fi
  local now elapsed
  now="$(date +%s)"
  elapsed=$(( now - TRANSFER_START_EPOCH ))
  (( elapsed < 1 )) && elapsed=1
  echo $(( TRANSFERRED_BYTES / elapsed ))
}

purge_stale_tmp_files() {
  local dir="$1"
  [[ -d "$dir" ]] || return 0
  find "$dir" -maxdepth 1 -type f \
    \( -name '.tmp.best.ckpt.*' -o -name '.tmp.best.info.json.*' -o -name '.tmp.best.ckpt.stderr.*' \) \
    -print0 2>/dev/null | while IFS= read -r -d '' f; do
      rm -f -- "$f"
    done
}

# Best-effort local ckpt validation:
# 1) file exists
# 2) size matches remote expected size
# 3) if it's a zip-format torch checkpoint, test zip integrity
validate_local_ckpt() {
  local path="$1"
  local expected_size="${2:-0}"

  [[ -f "$path" ]] || return 1

  local sz
  sz="$(file_size "$path")"
  [[ "$sz" =~ ^[0-9]+$ ]] || return 1
  (( sz > 0 )) || return 1

  if [[ "$expected_size" =~ ^[0-9]+$ ]] && (( expected_size > 0 )) && (( sz != expected_size )); then
    return 1
  fi

  if command -v python3 >/dev/null 2>&1; then
    python3 - "$path" <<'PY'
import sys, zipfile

p = sys.argv[1]
try:
    with open(p, 'rb') as f:
        sig = f.read(4)
except Exception:
    sys.exit(1)

# Newer torch checkpoints are often zip archives.
# If it's zip-like, test integrity.
if sig[:2] == b'PK':
    try:
        with zipfile.ZipFile(p) as zf:
            bad = zf.testzip()
        sys.exit(0 if bad is None else 1)
    except Exception:
        sys.exit(1)

# Non-zip checkpoint: size check only.
sys.exit(0)
PY
    return $?
  fi

  return 0
}

best_info_matches() {
  local info_path="$1"
  local run="$2"
  local remote_path="$3"
  local ckpt_name="$4"

  [[ -f "$info_path" ]] || return 1

  python3 - "$info_path" "$run" "$remote_path" "$ckpt_name" <<'PY'
import json, sys

info_path, run, remote_path, ckpt_name = sys.argv[1:]
try:
    with open(info_path, 'r', encoding='utf-8') as f:
        obj = json.load(f)
except Exception:
    sys.exit(1)

best = obj.get("best", {})
ok = (
    str(obj.get("run", "")) == run and
    str(best.get("remote_path", "")) == remote_path and
    str(best.get("ckpt_name", "")) == ckpt_name
)
sys.exit(0 if ok else 1)
PY
}

write_best_info() {
  local local_dir="$1"
  local task="$2"
  local model="$3"
  local run="$4"
  local remote_path="$5"
  local ckpt_name="$6"
  local epoch_s="$7"
  local loss_s="$8"

  local info_tmp
  info_tmp="$(mktemp --tmpdir="$local_dir" ".tmp.best.info.json.XXXXXX")"

  python3 - "$task" "$model" "$run" "$remote_path" "$ckpt_name" "$epoch_s" "$loss_s" > "$info_tmp" <<'PY'
import json, sys
task, model, run, remote_path, ckpt_name, epoch_s, loss_s = sys.argv[1:]
epoch = int(epoch_s) if epoch_s.strip() else None
val_loss = float(loss_s) if loss_s.strip() else None
obj = {
    "task": task,
    "model": model,
    "run": run,
    "best": {
        "epoch": epoch,
        "val_loss": val_loss,
        "remote_path": remote_path,
        "ckpt_name": ckpt_name
    }
}
print(json.dumps(obj, ensure_ascii=False, indent=2))
PY

  mv -f -- "$info_tmp" "${local_dir}/best.info.json"
}

# -----------------------------
# Interrupt cleanup
# -----------------------------
ACTIVE_PID=""
ACTIVE_TMP=""
LAST_ERROR=""

cleanup_on_signal() {
  clear_line
  log_err "Interrupted."
  if [[ -n "${ACTIVE_PID:-}" ]]; then
    kill "${ACTIVE_PID}" 2>/dev/null || true
  fi
  if [[ -n "${ACTIVE_TMP:-}" ]]; then
    rm -f -- "${ACTIVE_TMP}" 2>/dev/null || true
  fi
  exit 130
}
trap cleanup_on_signal INT TERM

# -----------------------------
# Fetch remote manifest
# Only BEST ckpt from latest run
# Fields:
# task, model, latest_run, remote_best, best_epoch, best_val_loss, best_ckpt_name, best_size
# -----------------------------
log
log "${C_BOLD}============================================================${C_RESET}"
log "${C_BOLD} Sync BEST checkpoints only${C_RESET}"
log "Mode  : $([[ $RESUME -eq 1 ]] && echo "resume" || echo "fresh")"
log "Remote: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}"
log "Local : ${LOCAL_BASE}"
log "${C_BOLD}============================================================${C_RESET}"
log

log_info "Scanning remote latest runs and best checkpoints..."

manifest="$(
  "${SSH_BASE[@]}" "${REMOTE_USER}@${REMOTE_HOST}" \
  "python3 - '$REMOTE_BASE' <<'PY'
import os, re, math, sys

base = sys.argv[1]
ts_re  = re.compile(r'^\\d{4}-\\d{2}-\\d{2}_\\d{2}-\\d{2}-\\d{2}$')
val_re = re.compile(r'val_loss=([0-9]*\\.?[0-9]+(?:[eE][-+]?\\d+)?)')
epo_re = re.compile(r'epoch=(\\d+)')

def isdir(p):
    return os.path.isdir(p)

if not isdir(base):
    raise SystemExit(f'[ERR] base path does not exist: {base}')

for task in sorted(os.listdir(base)):
    task_dir = os.path.join(base, task)
    if not isdir(task_dir):
        continue

    for model in sorted(os.listdir(task_dir)):
        model_dir = os.path.join(task_dir, model)
        if not isdir(model_dir):
            continue

        runs = [
            d for d in os.listdir(model_dir)
            if ts_re.match(d) and isdir(os.path.join(model_dir, d))
        ]
        if not runs:
            continue

        latest_run = max(runs)
        ckpt_dir = os.path.join(model_dir, latest_run, 'checkpoints')

        remote_best = ''
        best_loss = math.inf
        best_epoch = ''
        best_ckpt_name = ''

        if isdir(ckpt_dir):
            for fn in os.listdir(ckpt_dir):
                if not (fn.startswith('epoch=') and fn.endswith('.ckpt') and 'val_loss=' in fn):
                    continue
                m = val_re.search(fn)
                if not m:
                    continue
                loss = float(m.group(1))
                if loss < best_loss:
                    best_loss = loss
                    remote_best = os.path.realpath(os.path.join(ckpt_dir, fn))
                    best_ckpt_name = fn
                    me = epo_re.search(fn)
                    best_epoch = me.group(1) if me else ''

        if not remote_best or not os.path.exists(remote_best):
            print(f'[WARN] no best checkpoint: {task}/{model} (latest run: {latest_run})', file=sys.stderr)
            continue

        best_size = os.path.getsize(remote_best)
        best_loss_str = '' if not math.isfinite(best_loss) else repr(best_loss)

        print(
            f'{task}\\t{model}\\t{latest_run}\\t{remote_best}\\t'
            f'{best_epoch}\\t{best_loss_str}\\t{best_ckpt_name}\\t{best_size}'
        )
PY"
)"

mapfile -t JOBS < <(printf '%s\n' "$manifest" | sed '/^[[:space:]]*$/d')

TOTAL_ITEMS=${#JOBS[@]}
TOTAL_BYTES=0
for line in "${JOBS[@]}"; do
  IFS=$'\t' read -r task model latest_run remote_best best_epoch best_loss best_ckpt_name best_size <<< "$line"
  [[ -n "${best_size:-}" ]] && TOTAL_BYTES=$((TOTAL_BYTES + best_size))
done

if (( TOTAL_ITEMS == 0 )); then
  log_warn "No best checkpoints found."
  exit 0
fi

log_ok "Found ${TOTAL_ITEMS} best checkpoint(s), total size: $(human_bytes "$TOTAL_BYTES")"
log

GLOBAL_START_EPOCH="$(date +%s)"
TRANSFER_START_EPOCH=""
DONE_BYTES=0
TRANSFERRED_BYTES=0

SKIPPED_COUNT=0
DOWNLOADED_COUNT=0
REDOWNLOADED_COUNT=0
FAIL_COUNT=0

download_one() {
  local remote_path="$1"
  local local_path="$2"
  local expected_size="${3:-0}"
  local item_idx="${4:-0}"
  local item_total="${5:-0}"
  local label="${6:-}"

  LAST_ERROR=""

  if [[ -z "${remote_path//[[:space:]]/}" ]]; then
    LAST_ERROR="empty remote path"
    return 1
  fi

  local local_dir
  local_dir="$(dirname -- "$local_path")"
  mkdir -p -- "$local_dir"

  local tmp errfile
  tmp="$(mktemp --tmpdir="$local_dir" ".tmp.$(basename -- "$local_path").XXXXXX")"
  errfile="$(mktemp --tmpdir="$local_dir" ".tmp.$(basename -- "$local_path").stderr.XXXXXX")"

  ACTIVE_TMP="$tmp"

  if [[ -z "${TRANSFER_START_EPOCH:-}" ]]; then
    TRANSFER_START_EPOCH="$(date +%s)"
  fi

  "${SCP_BASE[@]}" -- "${REMOTE_USER}@${REMOTE_HOST}:${remote_path}" "$tmp" > /dev/null 2>"$errfile" &
  ACTIVE_PID=$!

  local cur=0 now elapsed speed total_done actual_size
  while kill -0 "$ACTIVE_PID" 2>/dev/null; do
    cur=0
    if [[ -e "$tmp" ]]; then
      cur="$(file_size "$tmp")"
    fi

    now="$(date +%s)"
    elapsed=$(( now - TRANSFER_START_EPOCH ))
    (( elapsed < 1 )) && elapsed=1
    speed=$(( (TRANSFERRED_BYTES + cur) / elapsed ))
    total_done=$(( DONE_BYTES + cur ))

    render_progress "$total_done" "$TOTAL_BYTES" "$speed" "$item_idx" "$item_total" "$label"
    sleep 0.5
  done

  if wait "$ACTIVE_PID"; then
    actual_size="$(file_size "$tmp")"
    if [[ "$expected_size" =~ ^[0-9]+$ ]] && (( expected_size > 0 )) && (( actual_size != expected_size )); then
      LAST_ERROR="size mismatch after download: got=${actual_size}, expected=${expected_size}"
      rm -f -- "$tmp" "$errfile"
      ACTIVE_PID=""
      ACTIVE_TMP=""
      return 1
    fi

    mv -f -- "$tmp" "$local_path"
    rm -f -- "$errfile"
    ACTIVE_PID=""
    ACTIVE_TMP=""
    return 0
  else
    if [[ -s "$errfile" ]]; then
      LAST_ERROR="$(tr '\n' ' ' < "$errfile" | sed 's/[[:space:]]\+/ /g')"
    else
      LAST_ERROR="scp failed"
    fi
    rm -f -- "$tmp" "$errfile"
    ACTIVE_PID=""
    ACTIVE_TMP=""
    return 1
  fi
}

for ((i=0; i<TOTAL_ITEMS; i++)); do
  line="${JOBS[$i]}"
  IFS=$'\t' read -r task model latest_run remote_best best_epoch best_loss best_ckpt_name best_size <<< "$line"

  label="${task}/${model}"
  local_dir="${LOCAL_BASE}/${task}/${model}/checkpoints"
  local_best_path="${local_dir}/best.ckpt"
  local_info_path="${local_dir}/best.info.json"

  mkdir -p -- "$local_dir"
  purge_stale_tmp_files "$local_dir"

  log_info "[$((i + 1))/${TOTAL_ITEMS}] ${label}"

  redownload_reason=""
  status="downloaded"

  if (( RESUME )); then
    metadata_ok=1
    if [[ -f "$local_info_path" ]]; then
      if best_info_matches "$local_info_path" "$latest_run" "$remote_best" "$best_ckpt_name"; then
        metadata_ok=1
      else
        metadata_ok=0
      fi
    else
      metadata_ok=2
    fi

    if validate_local_ckpt "$local_best_path" "${best_size:-0}"; then
      if (( metadata_ok == 0 )); then
        rm -f -- "$local_best_path" "$local_info_path"
        redownload_reason="stale_metadata"
        log_warn "Local file exists but metadata does not match current remote best; removed and will re-download."
      else
        DONE_BYTES=$((DONE_BYTES + best_size))
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        write_best_info "$local_dir" "$task" "$model" "$latest_run" "$remote_best" "$best_ckpt_name" "$best_epoch" "$best_loss"

        avg_speed="$(calc_avg_speed)"
        render_progress "$DONE_BYTES" "$TOTAL_BYTES" "$avg_speed" "$((i + 1))" "$TOTAL_ITEMS" "$label"
        printf '\n'

        log_ok "Already complete, skipped: ${local_best_path} ($(human_bytes "$best_size"))"

        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
          "$task" "$model" "best" "$local_best_path" "$remote_best" "$latest_run" \
          "${best_epoch:-}" "${best_loss:-}" "${best_ckpt_name:-}" "${best_size:-0}" "skipped_existing" \
          >> "$summary_tsv"

        log
        continue
      fi
    elif [[ -e "$local_best_path" ]]; then
      local_bad_size="$(file_size "$local_best_path")"
      rm -f -- "$local_best_path" "$local_info_path"
      redownload_reason="bad_local"
      log_warn "Found incomplete/corrupted local file, removed: ${local_best_path} (local=$(human_bytes "$local_bad_size"), remote=$(human_bytes "$best_size"))"
    fi
  fi

  if [[ -n "$redownload_reason" ]]; then
    status="redownloaded"
  fi

  if download_one "$remote_best" "$local_best_path" "${best_size:-0}" "$((i + 1))" "$TOTAL_ITEMS" "$label"; then
    downloaded_size="$(file_size "$local_best_path")"
    DONE_BYTES=$((DONE_BYTES + downloaded_size))
    TRANSFERRED_BYTES=$((TRANSFERRED_BYTES + downloaded_size))

    write_best_info "$local_dir" "$task" "$model" "$latest_run" "$remote_best" "$best_ckpt_name" "$best_epoch" "$best_loss"

    avg_speed="$(calc_avg_speed)"
    render_progress "$DONE_BYTES" "$TOTAL_BYTES" "$avg_speed" "$((i + 1))" "$TOTAL_ITEMS" "$label"
    printf '\n'

    if [[ "$status" == "redownloaded" ]]; then
      REDOWNLOADED_COUNT=$((REDOWNLOADED_COUNT + 1))
      log_ok "Re-downloaded best.ckpt -> ${local_best_path} ($(human_bytes "$downloaded_size"))"
    else
      DOWNLOADED_COUNT=$((DOWNLOADED_COUNT + 1))
      log_ok "Downloaded best.ckpt -> ${local_best_path} ($(human_bytes "$downloaded_size"))"
    fi

    log_ok "Saved best.info.json | epoch=${best_epoch:-N/A}, val_loss=${best_loss:-N/A}"

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$task" "$model" "best" "$local_best_path" "$remote_best" "$latest_run" \
      "${best_epoch:-}" "${best_loss:-}" "${best_ckpt_name:-}" "${downloaded_size:-0}" "$status" \
      >> "$summary_tsv"
  else
    clear_line
    FAIL_COUNT=$((FAIL_COUNT + 1))
    log_warn "Failed to download ${label}: ${LAST_ERROR:-unknown error}"

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$task" "$model" "best" "$local_best_path" "$remote_best" "$latest_run" \
      "${best_epoch:-}" "${best_loss:-}" "${best_ckpt_name:-}" "${best_size:-0}" "failed" \
      >> "$summary_tsv"
  fi

  log
done

elapsed_total=$(( $(date +%s) - GLOBAL_START_EPOCH ))
(( elapsed_total < 1 )) && elapsed_total=1
avg_speed="$(calc_avg_speed)"
READY_COUNT=$((SKIPPED_COUNT + DOWNLOADED_COUNT + REDOWNLOADED_COUNT))

log "${C_BOLD}============================================================${C_RESET}"
log "${C_BOLD}Done${C_RESET}"
log "Already OK : ${SKIPPED_COUNT}"
log "Downloaded : ${DOWNLOADED_COUNT}"
log "Redownload : ${REDOWNLOADED_COUNT}"
log "Failed     : ${FAIL_COUNT}"
log "Ready now  : ${READY_COUNT} / ${TOTAL_ITEMS}"
log "Available  : $(human_bytes "$DONE_BYTES") / $(human_bytes "$TOTAL_BYTES")"
log "Transferred: $(human_bytes "$TRANSFERRED_BYTES")"
log "Elapsed    : $(format_duration "$elapsed_total")"
log "Avg SPD    : $(human_bytes "$avg_speed")/s"
log "Output     : ${LOCAL_BASE}"
log "Summary    : ${summary_tsv}"
log "${C_BOLD}============================================================${C_RESET}"

log "---- summary preview ----"
if command -v column >/dev/null 2>&1; then
  column -ts $'\t' "$summary_tsv" | tail -n 50 || true
else
  tail -n 50 "$summary_tsv" || true
fi

# ./syn_ckpts.sh
# ./syn_ckpts.sh --resume
