#!/bin/bash
# Full revision execution script with budget guards and checkpointing
# Strategy:
#   1. T14 (PubMed PM2.5)  — ~$2  — validates pipeline + smallest cost
#   2. T1 humaneval        — ~$3
#   3. T1 gsm8k            — ~$3
#   4. T4 multi-turn ext.  — ~$6 (gpt-4-turbo + deepseek)
#   Total: ~$14 USD vs $50 budget ($36 USD reserve)

set -e
cd /Users/lucasrover/paper-experiment

PYTHON=.venv/bin/python
OUT=outputs/revision/runs
CHECKPOINT=outputs/revision/checkpoint.json
LOG_DIR=outputs/revision/logs
mkdir -p "$OUT" "$LOG_DIR"

set -a
source /Users/lucasrover/.env
set +a

echo "================================================================"
echo "  NatComms Revision — Experiment Execution"
echo "  Started: $(date)"
echo "  Budget cap: \$50 USD"
echo "  Checkpoint: $CHECKPOINT"
echo "================================================================"

# Stacks to use:
#   - All 3 local (free, fast)
#   - together-llama3 (cheap, quasi-isolation)
#   - claude-sonnet-4-5 (R3 needs)
#   - gpt-4o (current OpenAI snapshot, D8 cheap)
#   - gemini-2-5-pro (R3 needs)
#   - deepseek-chat (R3 needs)
# Skip: gpt-4 (expensive, gpt-4o serves D8), gpt-4-turbo (gpt-4o cheaper),
#       claude-opus-4-7 (reserved for T3 LLM-judge)

STACKS=(
  "llama3-8b-local"
  "mistral-7b-local"
  "gemma2-9b-local"
  "together-llama3"
  "claude-sonnet-4-5"
  "gpt-4o"
  "gemini-2-5-pro"
  "deepseek-chat"
)

run_task() {
  local task=$1
  local n_problems=$2
  local n_reps=$3
  local desc=$4

  echo ""
  echo "================================================================"
  echo "  PHASE: $desc"
  echo "  Task: $task | n_problems: $n_problems | n_reps: $n_reps"
  echo "  Started: $(date)"
  echo "================================================================"

  for stack in "${STACKS[@]}"; do
    echo ""
    echo "[$desc] Stack: $stack"
    LOG_FILE="$LOG_DIR/${task}_${stack}.log"
    $PYTHON run_revision_experiments.py \
      --task "$task" \
      --stack "$stack" \
      --condition C1 \
      --n-problems "$n_problems" \
      --n-reps "$n_reps" \
      --output-dir "$OUT" \
      --checkpoint "$CHECKPOINT" \
      --budget-usd 50 \
      --resume \
      --execute 2>&1 | tee "$LOG_FILE" | tail -20
    echo "  → log: $LOG_FILE"
  done
}

# Phase 1: T14 PubMed (smallest, validates pipeline)
run_task "pubmed_pm25" 10 5 "T14 — 10 PubMed PM2.5 abstracts × 5 reps"

# Phase 2: T1 HumanEval (code)
run_task "humaneval" 30 5 "T1 — 30 HumanEval code problems × 5 reps"

# Phase 3: T1 GSM8K (math)
run_task "gsm8k" 30 5 "T1 — 30 GSM8K math problems × 5 reps"

# Phase 4: T4 Multi-turn extension (only API stacks per R3)
# Skip the "for all stacks" loop — only run multi-turn on stacks not yet covered
echo ""
echo "================================================================"
echo "  PHASE: T4 — Multi-turn extension to GPT-4o + DeepSeek (R3 item 3)"
echo "================================================================"
for stack in "gpt-4o" "deepseek-chat"; do
  echo ""
  echo "[T4] Stack: $stack"
  LOG_FILE="$LOG_DIR/multiturn_${stack}.log"
  $PYTHON run_revision_experiments.py \
    --task multiturn_extension \
    --stack "$stack" \
    --condition C1 \
    --n-problems 10 \
    --n-reps 5 \
    --output-dir "$OUT" \
    --checkpoint "$CHECKPOINT" \
    --budget-usd 50 \
    --resume \
    --execute 2>&1 | tee "$LOG_FILE" | tail -20
done

echo ""
echo "================================================================"
echo "  EXECUTION COMPLETE"
echo "  Finished: $(date)"
echo "  Checkpoint: $CHECKPOINT"
echo "  Run outputs: $OUT"
echo "================================================================"
