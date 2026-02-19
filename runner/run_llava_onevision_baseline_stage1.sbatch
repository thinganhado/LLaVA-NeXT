#!/usr/bin/env bash
#SBATCH --job-name=llava_ov72b_stage1
#SBATCH --partition=h24gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/llava_ov72b_stage1_%j.out
#SBATCH --account=OD-237777
#SBATCH --gres=gpu:4
#SBATCH --exclusive

set -euo pipefail

if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi

CONDA_ENV_NAME="${CONDA_ENV_NAME:-deepfake}"
conda activate "${CONDA_ENV_NAME}"

if [[ -n "${SCRIPT_DIR_OVERRIDE:-}" ]]; then
  SCRIPT_DIR="${SCRIPT_DIR_OVERRIDE}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_PATH="${SLURM_JOB_SCRIPT:-${BASH_SOURCE[0]:-$0}}"
  SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
fi

REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || (cd "${SCRIPT_DIR}/.." && pwd))"
cd "${REPO_ROOT}"
mkdir -p logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256

# Defaults can be overridden at submit time, e.g.:
# sbatch --export=ALL,NUM_SHARDS=8,SHARD_ID=0 scripts/run_llava_onevision_baseline_stage1.sbatch
MODEL_ID="${MODEL_ID:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/VLM/llava-onevision-qwen2-72b-si-hf/}"
META_JSON="${META_JSON:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_SFT/stage1_val.json}"
META_CSV="${META_CSV:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_SFT/stage1_gt.csv}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/img/specs/grid}"
SYSTEM_FILE="${SYSTEM_FILE:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_prompts/baseline_system.txt}"
USER_TEMPLATE_FILE="${USER_TEMPLATE_FILE:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_prompts/baseline_user.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_strongVLM/}"

# Default to fp16 to avoid bf16/fp16 mismatch issues in some environments.
DTYPE="${DTYPE:-float16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
IMAGE_MAX_SIDE="${IMAGE_MAX_SIDE:-896}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-0}"
DO_SAMPLE="${DO_SAMPLE:-0}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
ATTN_IMPL="${ATTN_IMPL:-}"

NUM_SHARDS="${NUM_SHARDS:-1}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  SHARD_ID="${SHARD_ID:-${SLURM_ARRAY_TASK_ID}}"
else
  SHARD_ID="${SHARD_ID:-0}"
fi
if [[ "${NUM_SHARDS}" -gt 1 ]]; then
  OUTPUT_JSONL="${OUTPUT_JSONL:-${OUTPUT_DIR%/}/llava_onevision_outputs_shard_${SHARD_ID}.jsonl}"
else
  OUTPUT_JSONL="${OUTPUT_JSONL:-${OUTPUT_DIR%/}/llava_onevision_outputs.jsonl}"
fi

RUNNER_PY=""
if [[ -f "${REPO_ROOT}/runner/llava_onevision_baseline_runner.py" ]]; then
  RUNNER_PY="${REPO_ROOT}/runner/llava_onevision_baseline_runner.py"
elif [[ -f "${REPO_ROOT}/scripts/llava_onevision_baseline_runner.py" ]]; then
  RUNNER_PY="${REPO_ROOT}/scripts/llava_onevision_baseline_runner.py"
else
  echo "[ERR] Cannot find llava_onevision_baseline_runner.py under runner/ or scripts/ in ${REPO_ROOT}" >&2
  exit 1
fi

CMD=(
  python "${RUNNER_PY}"
  --model-id "${MODEL_ID}"
  --meta-json "${META_JSON}"
  --meta-csv "${META_CSV}"
  --image-folder "${IMAGE_FOLDER}"
  --system-file "${SYSTEM_FILE}"
  --user-template-file "${USER_TEMPLATE_FILE}"
  --output-dir "${OUTPUT_DIR}"
  --output-jsonl "${OUTPUT_JSONL}"
  --dtype "${DTYPE}"
  --device-map "${DEVICE_MAP}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --image-max-side "${IMAGE_MAX_SIDE}"
  --num-shards "${NUM_SHARDS}"
  --shard-id "${SHARD_ID}"
)

if [[ "${DO_SAMPLE}" == "1" ]]; then
  CMD+=(--do-sample --temperature "${TEMPERATURE}" --top-p "${TOP_P}")
fi

if [[ -n "${ATTN_IMPL}" ]]; then
  CMD+=(--attn-implementation "${ATTN_IMPL}")
fi
if [[ "${LOAD_IN_4BIT}" == "1" ]]; then
  CMD+=(--load-in-4bit)
fi

echo "[info] running command:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
