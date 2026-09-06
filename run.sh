#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_NAME="${MODEL_NAME:-deepseek-v4-flash}"

(
  cd "$REPOSITORY_ROOT/study1/nudge_replication"
  python run.py --model_name "$MODEL_NAME"
  python run_long_term_0.py --model_name "$MODEL_NAME"
  python run_long_term_2.py --model_name "$MODEL_NAME"
  python run_long_term_3.py --model_name "$MODEL_NAME"
)

(
  cd "$REPOSITORY_ROOT/study2/human_experiments"
  python run_human.py --model_name "$MODEL_NAME"
)

(
  cd "$REPOSITORY_ROOT/study3/longitudinal_simulation"
  python run_long.py --model_name "$MODEL_NAME"
  python run_freq.py --model_name "$MODEL_NAME"
  python run_freq.py --frequency 6 --model_name "$MODEL_NAME"
)

echo "All main experiments finished successfully."
