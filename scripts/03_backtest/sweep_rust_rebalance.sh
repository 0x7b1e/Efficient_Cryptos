#!/usr/bin/env bash
set -euo pipefail

# Run the Rust factor backtest over several rebalance frequencies.  The input
# predictions are 30-minute bars, so the frequencies below map to these bar
# intervals: 1h=2, 2h=4, 4h=8, 8h=16, 12h=24, 24h=48.

usage() {
  cat >&2 <<'EOF'
Usage:
  scripts/sweep_rust_rebalance.sh \
    --predictions-dir outputs/rust_predictions_train_2024_2025 \
    --output-root outputs/rust_sweep_train_2024_2025 \
    [--jobs 2] [--long-percent 10] [--short-percent 10] \
    [--gross-leverage 1] [--plot-max-points 5000] [--no-plots] [--force]
EOF
}

PREDICTIONS_DIR="${PREDICTIONS_DIR:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
JOBS="${JOBS:-2}"
LONG_PERCENT="${LONG_PERCENT:-10}"
SHORT_PERCENT="${SHORT_PERCENT:-10}"
GROSS_LEVERAGE="${GROSS_LEVERAGE:-1}"
MAKE_PLOTS="${MAKE_PLOTS:-1}"
FORCE="${FORCE:-0}"
PLOT_MAX_POINTS="${PLOT_MAX_POINTS:-5000}"
RUST_BIN="${RUST_BIN:-}"
INTERNAL_TASK=0
TASK_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --predictions-dir) PREDICTIONS_DIR=$2; shift 2 ;;
    --output-root) OUTPUT_ROOT=$2; shift 2 ;;
    --jobs) JOBS=$2; shift 2 ;;
    --long-percent) LONG_PERCENT=$2; shift 2 ;;
    --short-percent) SHORT_PERCENT=$2; shift 2 ;;
    --gross-leverage) GROSS_LEVERAGE=$2; shift 2 ;;
    --plot-max-points) PLOT_MAX_POINTS=$2; shift 2 ;;
    --rust-bin) RUST_BIN=$2; shift 2 ;;
    --task) INTERNAL_TASK=1; shift; TASK_ARGS=("$@"); break ;;
    --no-plots) MAKE_PLOTS=0; shift ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$PREDICTIONS_DIR" || -z "$OUTPUT_ROOT" ]]; then
  usage
  exit 2
fi
if ! [[ "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "--jobs must be a positive integer" >&2
  exit 2
fi

if [[ -z "$RUST_BIN" ]]; then
  RUST_BIN="$(pwd)/rust_backtest/target/release/crypto-factor-backtest"
fi
PLOT_SCRIPT="$(pwd)/scripts/03_backtest/plot_rust_backtest.py"
# Prefer the project virtualenv, fall back to the python on PATH.
PYTHON_BIN="${PYTHON_BIN:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || command -v python3 )}"

if [[ ! -x "$RUST_BIN" ]]; then
  echo "Rust binary not found or not executable: $RUST_BIN" >&2
  exit 1
fi
if [[ ! -d "$PREDICTIONS_DIR" ]]; then
  echo "prediction directory not found: $PREDICTIONS_DIR" >&2
  exit 1
fi

run_task() {
  local input=$1
  local frequency=$2
  local bars=$3
  local factor
  factor=$(basename "$input" .csv)

  local result_dir="$OUTPUT_ROOT/$frequency"
  local metrics="$result_dir/${factor}.metrics.json"
  local layers="$result_dir/${factor}.layers.csv"
  local timeseries="$result_dir/${factor}.timeseries.csv"
  local equity_plot="$result_dir/plots/${factor}.equity.png"
  local layers_plot="$result_dir/plots/${factor}.layers.png"
  local log="$result_dir/logs/${factor}.log"
  mkdir -p "$result_dir/plots" "$result_dir/logs"

  if [[ "$FORCE" -eq 0 && -s "$metrics" && -s "$layers" && -s "$timeseries" && ( "$MAKE_PLOTS" -eq 0 || ( -s "$equity_plot" && -s "$layers_plot" ) ) ]]; then
    echo "SKIP $frequency $factor"
    return 0
  fi

  {
    echo "START frequency=$frequency bars=$bars factor=$factor"
    "$RUST_BIN" \
      --input "$input" \
      --output "$metrics" \
      --layers-output "$layers" \
      --timeseries-output "$timeseries" \
      --long-percent "$LONG_PERCENT" \
      --short-percent "$SHORT_PERCENT" \
      --signal-interval-bars "$bars" \
      --signal-hold-bars "$bars" \
      --gross-leverage "$GROSS_LEVERAGE" \
      --include-costs \
      --include-funding

    if [[ "$MAKE_PLOTS" -eq 1 ]]; then
      "$PYTHON_BIN" "$PLOT_SCRIPT" \
        --metrics "$metrics" \
        --layers "$layers" \
        --timeseries "$timeseries" \
        --kind equity \
        --max-points "$PLOT_MAX_POINTS" \
        --out "$equity_plot"
      "$PYTHON_BIN" "$PLOT_SCRIPT" \
        --metrics "$metrics" \
        --layers "$layers" \
        --timeseries "$timeseries" \
        --kind layers \
        --max-points "$PLOT_MAX_POINTS" \
        --out "$layers_plot"
    fi
    echo "DONE frequency=$frequency bars=$bars factor=$factor"
  } >"$log" 2>&1
  echo "DONE $frequency $factor"
}

# Internal mode used by xargs.  Keeping the task in this same script avoids
# fragile shell quoting while still allowing independent factor/frequency jobs.
if [[ "$INTERNAL_TASK" -eq 1 ]]; then
  run_task "${TASK_ARGS[@]}"
  exit 0
fi

mapfile -t INPUTS < <(find "$PREDICTIONS_DIR" -maxdepth 1 -type f -name '*.csv' | sort)
if [[ "${#INPUTS[@]}" -eq 0 ]]; then
  echo "no prediction CSV files found in $PREDICTIONS_DIR" >&2
  exit 1
fi

TASKS=$(mktemp)
trap 'rm -f "$TASKS"' EXIT
for input in "${INPUTS[@]}"; do
  printf '%s\0%s\0%s\0' "$input" "1h" "2" >>"$TASKS"
  printf '%s\0%s\0%s\0' "$input" "2h" "4" >>"$TASKS"
  printf '%s\0%s\0%s\0' "$input" "4h" "8" >>"$TASKS"
  printf '%s\0%s\0%s\0' "$input" "8h" "16" >>"$TASKS"
  printf '%s\0%s\0%s\0' "$input" "12h" "24" >>"$TASKS"
  printf '%s\0%s\0%s\0' "$input" "24h" "48" >>"$TASKS"
done

total=$(( ${#INPUTS[@]} * 6 ))
echo "Running ${#INPUTS[@]} factors across 6 rebalance frequencies: $total jobs, workers=$JOBS"
export PREDICTIONS_DIR OUTPUT_ROOT JOBS LONG_PERCENT SHORT_PERCENT GROSS_LEVERAGE MAKE_PLOTS FORCE PLOT_MAX_POINTS RUST_BIN PLOT_SCRIPT
xargs -0 -r -n 3 -P "$JOBS" "$0" --task <"$TASKS"
echo "Sweep complete: $OUTPUT_ROOT"
