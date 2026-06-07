#!/bin/bash
# sc-3031 Phase C: A/B the new Rust adapter vs the old Python Mlx*Adapter, matched
# (model, prompt, seed, dims), across the txt2img families. Records metrics to a TSV.
set -u
REPO=/Users/michael/Repos/SceneWorks
cd "$REPO" || exit 1
AB=/tmp/sc3031_ab
MAIN="$HOME/Library/Application Support/SceneWorks/python/venv/bin/python"
LOG="$AB/driver.log"; TSV="$AB/ab_metrics.tsv"
: > "$LOG"; printf 'model\tmetrics\n' > "$TSV"

export SCENEWORKS_DATA_DIR="$HOME/Library/Application Support/SceneWorks/data"
export SCENEWORKS_CONFIG_DIR="$HOME/Library/Application Support/SceneWorks/config"
export HF_HOME="$HOME/.cache/huggingface"
export SCENEWORKS_MLX_FLUX_PYTHON="$HOME/Library/Application Support/SceneWorks/python/mlx-flux-venv/bin/python"
export PYTHONPATH="$REPO/apps/worker:$REPO/packages/shared"
PROMPT="a calm ocean wave at sunset, cinematic"

models=(flux_schnell flux_dev qwen_image flux2_klein_9b flux2_klein_9b_kv)

for m in "${models[@]}"; do
  payload="{\"projectId\":\"ab\",\"model\":\"$m\",\"prompt\":\"$PROMPT\",\"seed\":42,\"width\":512,\"height\":512,\"count\":1}"
  echo "[$(date '+%H:%M:%S')] ==== $m ====" | tee -a "$LOG"

  echo "[$(date '+%H:%M:%S')] rust dump $m" | tee -a "$LOG"
  SC3031_PAYLOAD="$payload" SC3031_OUT="$AB/rust_$m.png" \
    cargo test -p sceneworks-worker --lib -- --ignored --exact \
    image_jobs::tests::sc3031_ab_dump_txt2img --nocapture >> "$LOG" 2>&1
  rrc=$?

  echo "[$(date '+%H:%M:%S')] python dump $m" | tee -a "$LOG"
  SC3031_PAYLOAD="$payload" SC3031_OUT="$AB/py_$m.png" \
    "$MAIN" "$AB/ab_python.py" >> "$LOG" 2>&1
  prc=$?

  if [ $rrc -eq 0 ] && [ $prc -eq 0 ] && [ -f "$AB/rust_$m.png" ] && [ -f "$AB/py_$m.png" ]; then
    metrics=$("$MAIN" "$AB/compare.py" "$AB/rust_$m.png" "$AB/py_$m.png" 2>>"$LOG")
  else
    metrics="ERROR rust_rc=$rrc py_rc=$prc"
  fi
  printf '%s\t%s\n' "$m" "$metrics" | tee -a "$TSV"
done

echo "[$(date '+%H:%M:%S')] ===== DONE =====" | tee -a "$LOG"
echo "----- METRICS -----" | tee -a "$LOG"
cat "$TSV" | tee -a "$LOG"
