#!/usr/bin/env bash
# Internal implementation for the three user-facing manual MSMU stage scripts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

stage="${1:-}"
if [[ -z "${stage}" ]]; then
  echo "[msmu-manual] internal error: missing stage" >&2
  exit 2
fi
shift

print_models() {
  cat <<'EOF'
gpt5
gemini31pro
llava_next_mistral_7b
llava_next_yi_34b
internvl3_8b
internvl3_38b
internvl3_78b
qwen25_vl_base
qwen25_vl_32b
qwen25_vl_72b
qwen25_vl_peft
qwen3_vl_2b
qwen3_vl_4b
qwen3_vl_8b
qwen3_vl_32b
ssr
ssr_native
spatialrgpt
3dthinker
3dthinker_native
spatialbot
spatialbot_native
EOF
}

usage() {
  local command
  case "${stage}" in
    1) command="run_manual_stage1.sh MODEL [serve|check|run]" ;;
    2) command="run_manual_stage2.sh MODEL [serve|run]" ;;
    3) command="run_manual_stage3.sh MODEL [serve|infer|score]" ;;
    *) command="run_manual_stageN.sh MODEL ACTION" ;;
  esac
  cat <<EOF
Usage: bash scripts/msmu/${command}

Options:
  --list    list accepted MODEL names
  --help    show this help

The script loads .env.server itself. Set MANUAL_DRY_RUN=1 to print commands
without starting a model, using a GPU, or calling an API.
EOF
  if [[ "${stage}" == "3" ]]; then
    cat <<'EOF'

Stage 3 also accepts:
  bash scripts/msmu/run_manual_stage3.sh judge serve
EOF
  fi
}

case "${1:-}" in
  --help|-h)
    usage
    exit 0
    ;;
  --list)
    print_models
    if [[ "${stage}" == "3" ]]; then printf '%s\n' judge; fi
    exit 0
    ;;
  "")
    usage >&2
    exit 2
    ;;
esac

model="$1"
action="${2:-}"
if (( $# > 2 )); then
  usage >&2
  exit 2
fi

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/prepare_manual_test.sh"

manual_dry_run="${MANUAL_DRY_RUN:-0}"

run_command() {
  if [[ "${manual_dry_run}" == "1" ]]; then
    printf '[msmu-manual] dry-run:'
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

run_logged_command() {
  local log_path="$1"
  shift
  mkdir -p "$(dirname "${log_path}")"
  if [[ "${manual_dry_run}" == "1" ]]; then
    printf '[msmu-manual] dry-run:'
    printf ' %q' "$@"
    printf ' 2>&1 | tee %q\n' "${log_path}"
    return 0
  fi
  set +e
  "$@" 2>&1 | tee "${log_path}"
  local command_status=${PIPESTATUS[0]}
  set -e
  return "${command_status}"
}

fail() {
  echo "[msmu-manual] $*" >&2
  exit 2
}

blocked() {
  echo "[msmu-manual] $*" >&2
  exit 4
}

require_value() {
  local variable="$1" value="$2"
  if [[ -z "${value}" ]]; then
    fail "${variable} is missing from .env.server"
  fi
}

require_api_key() {
  local variable="$1"
  if [[ "${manual_dry_run}" != "1" && -z "${!variable:-}" ]]; then
    fail "export ${variable} in this terminal before running the API model"
  fi
}

slugify() {
  local value="$1"
  value="$(printf '%s' "${value}" | tr -cs 'A-Za-z0-9._-' '-')"
  value="${value#-}"
  value="${value%-}"
  printf '%s' "${value:-checkpoint}"
}

model_kind=""
profile=""
run_slug=""
model_path=""
served_model_name=""
serve_script=""
default_devices="0"
api_key_variable=""
qwen_model=""
qwen_revision=""
qwen_device_map="single"
qwen_batch_size="8"
qwen_min_free_gpu_mib="30000"

api_backend="${MANUAL_API_BACKEND:-openrouter}"
case "${model}" in
  gpt5)
    model_kind="api"
    profile="gpt5"
    case "${api_backend}" in
      openrouter) api_key_variable="OPENROUTER_API_KEY" ;;
      openai) api_key_variable="OPENAI_API_KEY" ;;
      *) fail "gpt5 supports MANUAL_API_BACKEND=openrouter or openai" ;;
    esac
    run_slug="gpt5-${api_backend}"
    ;;
  gemini31pro)
    model_kind="api"
    profile="gemini31pro"
    case "${api_backend}" in
      openrouter) api_key_variable="OPENROUTER_API_KEY" ;;
      google) api_key_variable="GEMINI_API_KEY" ;;
      *) fail "gemini31pro supports MANUAL_API_BACKEND=openrouter or google" ;;
    esac
    run_slug="gemini31pro-${api_backend}"
    ;;
  llava_next_mistral_7b)
    model_kind="vllm"
    profile="${model}"
    run_slug="llava-next-mistral-7b-vllm"
    model_path="${LLAVA_MISTRAL_7B_MODEL:-}"
    served_model_name="llava-next-mistral-7b-msmu"
    serve_script="${SCRIPT_DIR}/serve_llava_next.sh"
    default_devices="0"
    ;;
  llava_next_yi_34b)
    model_kind="vllm"
    profile="${model}"
    run_slug="llava-next-yi-34b-vllm"
    model_path="${LLAVA_YI_34B_MODEL:-}"
    served_model_name="llava-next-yi-34b-msmu"
    serve_script="${SCRIPT_DIR}/serve_llava_next.sh"
    default_devices="0,1"
    ;;
  internvl3_8b)
    model_kind="vllm"
    profile="${model}"
    run_slug="internvl3-8b-vllm"
    model_path="${INTERNVL3_8B_MODEL:-}"
    served_model_name="internvl3-8b-msmu"
    serve_script="${SCRIPT_DIR}/serve_internvl3.sh"
    default_devices="0"
    ;;
  internvl3_38b)
    model_kind="vllm"
    profile="${model}"
    run_slug="internvl3-38b-vllm"
    model_path="${INTERNVL3_38B_MODEL:-}"
    served_model_name="internvl3-38b-msmu"
    serve_script="${SCRIPT_DIR}/serve_internvl3.sh"
    default_devices="0,1"
    ;;
  internvl3_78b)
    model_kind="vllm"
    profile="${model}"
    run_slug="internvl3-78b-vllm"
    model_path="${INTERNVL3_78B_MODEL:-}"
    served_model_name="internvl3-78b-msmu"
    serve_script="${SCRIPT_DIR}/serve_internvl3.sh"
    default_devices="0,1"
    ;;
  qwen25_vl_base)
    model_kind="qwen"
    profile="qwen25_vl_7b"
    run_slug="qwen25-vl-base"
    qwen_model="${QWEN_BASE_MODEL:-}"
    qwen_revision="${QWEN_BASE_REVISION:-}"
    default_devices="0"
    ;;
  qwen25_vl_32b)
    model_kind="qwen"
    profile="qwen25_vl_32b"
    run_slug="qwen25-vl-32b"
    qwen_model="${QWEN_32B_MODEL:-}"
    qwen_revision="${QWEN_32B_REVISION:-}"
    default_devices="0"
    qwen_batch_size="1"
    qwen_min_free_gpu_mib="75000"
    ;;
  qwen25_vl_72b)
    model_kind="qwen"
    profile="qwen25_vl_72b"
    run_slug="qwen25-vl-72b"
    qwen_model="${QWEN_72B_MODEL:-}"
    qwen_revision="${QWEN_72B_REVISION:-}"
    default_devices="0,1"
    qwen_device_map="balanced"
    qwen_batch_size="1"
    qwen_min_free_gpu_mib="75000"
    ;;
  qwen25_vl_peft)
    model_kind="qwen_peft"
    profile="qwen25_vl_7b"
    qwen_model="${QWEN_BASE_MODEL:-}"
    qwen_revision="${QWEN_BASE_REVISION:-}"
    require_value QWEN_PEFT_CHECKPOINT "${QWEN_PEFT_CHECKPOINT:-}"
    checkpoint_parent="$(basename "$(dirname "${QWEN_PEFT_CHECKPOINT}")")"
    checkpoint_name="$(basename "${QWEN_PEFT_CHECKPOINT}")"
    run_slug="qwen25-vl-peft-$(slugify "${checkpoint_parent}-${checkpoint_name}")"
    ;;
  qwen3_vl_2b)
    model_kind="qwen"
    profile="qwen3_vl_2b"
    run_slug="qwen3-vl-2b"
    qwen_model="${QWEN3_2B_MODEL:-}"
    qwen_revision="${QWEN3_2B_REVISION:-}"
    default_devices="0"
    ;;
  qwen3_vl_4b)
    model_kind="qwen"
    profile="qwen3_vl_4b"
    run_slug="qwen3-vl-4b"
    qwen_model="${QWEN3_4B_MODEL:-}"
    qwen_revision="${QWEN3_4B_REVISION:-}"
    default_devices="0"
    ;;
  qwen3_vl_8b)
    model_kind="qwen"
    profile="qwen3_vl_8b"
    run_slug="qwen3-vl-8b"
    qwen_model="${QWEN3_8B_MODEL:-}"
    qwen_revision="${QWEN3_8B_REVISION:-}"
    default_devices="0"
    ;;
  qwen3_vl_32b)
    model_kind="qwen"
    profile="qwen3_vl_32b"
    run_slug="qwen3-vl-32b"
    qwen_model="${QWEN3_32B_MODEL:-}"
    qwen_revision="${QWEN3_32B_REVISION:-}"
    default_devices="0"
    qwen_batch_size="1"
    qwen_min_free_gpu_mib="75000"
    ;;
  ssr)
    model_kind="ssr"
    profile="ssr"
    run_slug="ssr-rgb-only"
    ;;
  ssr_native)
    model_kind="ssr"
    profile="ssr_native"
    run_slug="ssr-native"
    ;;
  spatialrgpt)
    model_kind="spatialrgpt"
    run_slug="spatialrgpt-rgb-only"
    model_path="${SPATIALRGPT_MODEL:-}"
    ;;
  3dthinker)
    model_kind="3dthinker"
    profile="3dthinker"
    run_slug="3dthinker-fair"
    model_path="${THREEDTHINKER_MODEL:-}"
    ;;
  3dthinker_native)
    model_kind="3dthinker"
    profile="3dthinker_native"
    run_slug="3dthinker-native"
    model_path="${THREEDTHINKER_MODEL:-}"
    ;;
  spatialbot)
    model_kind="spatialbot"
    profile="spatialbot"
    run_slug="spatialbot-rgb-only"
    model_path="${SPATIALBOT_MODEL:-}"
    ;;
  spatialbot_native)
    model_kind="spatialbot"
    profile="spatialbot_native"
    run_slug="spatialbot-native"
    model_path="${SPATIALBOT_MODEL:-}"
    ;;
  judge)
    if [[ "${stage}" != "3" ]]; then fail "judge is only available in stage 3"; fi
    model_kind="judge"
    run_slug="judge"
    ;;
  *)
    echo "[msmu-manual] unsupported model: ${model}" >&2
    echo "[msmu-manual] run with --list to see accepted names" >&2
    exit 2
    ;;
esac

if [[ -n "${MANUAL_RUN_SLUG:-}" ]]; then
  if [[ ! "${MANUAL_RUN_SLUG}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    fail "MANUAL_RUN_SLUG may contain only letters, digits, dot, underscore, and hyphen"
  fi
  run_slug="${MANUAL_RUN_SLUG}"
fi

case "${stage}" in
  1) stage_dir="01_canary" ;;
  2) stage_dir="02_smoke8" ;;
  3) stage_dir="03_full987" ;;
  *) fail "unsupported stage: ${stage}" ;;
esac
run_name="${stage_dir}/${run_slug}"

inference_base_url="${MANUAL_INFERENCE_BASE_URL:-http://127.0.0.1:18081/v1}"
judge_base_url="${MANUAL_JUDGE_BASE_URL:-${JUDGE_BASE_URL:-http://127.0.0.1:18080/v1}}"

serve_vllm() {
  require_value MODEL_PATH "${model_path}"
  local check_dir="${OUTPUT_ROOT}/${stage_dir}/${run_slug}"
  local devices="${MANUAL_CUDA_VISIBLE_DEVICES:-${default_devices}}"
  run_logged_command "${check_dir}/vllm_serve.log" \
    env PROFILE="${profile}" MODEL_PATH="${model_path}" \
    CUDA_VISIBLE_DEVICES="${devices}" \
      bash "${serve_script}"
}

preflight_vllm() {
  require_value MODEL_PATH "${model_path}"
  local check_dir="${OUTPUT_ROOT}/${stage_dir}/${run_slug}"
  mkdir -p "${check_dir}"
  run_command env PROFILE="${profile}" MODEL_PATH="${model_path}" \
    PREFLIGHT_REPORT="${check_dir}/processor_preflight.json" \
      bash "${SCRIPT_DIR}/preflight_vllm_processor.sh"
}

check_vllm_canary() {
  local check_dir="${OUTPUT_ROOT}/${stage_dir}/${run_slug}"
  mkdir -p "${check_dir}"
  run_command env PROFILE="${profile}" SERVED_MODEL_NAME="${served_model_name}" \
    INFERENCE_BASE_URL="${inference_base_url}" \
    CANARY_REPORT="${check_dir}/vision_canary.json" \
      bash "${SCRIPT_DIR}/canary_vllm_vision.sh"
}

dry_run_internvl78_config() {
  local check_dir="${OUTPUT_ROOT}/${stage_dir}/${run_slug}"
  run_logged_command "${check_dir}/vllm_config_dry_run.log" \
    env PROFILE="${profile}" MODEL_PATH="${model_path}" DRY_RUN=1 \
      bash "${serve_script}"
}

qwen_gpu_preflight() {
  local devices="${MANUAL_CUDA_VISIBLE_DEVICES:-${default_devices}}"
  run_command env CUDA_VISIBLE_DEVICES="${devices}" MIN_FREE_GPU_MIB="${qwen_min_free_gpu_mib}" \
    bash "${SCRIPT_DIR}/gpu_preflight.sh"
}

run_model_pipeline() {
  local target_mode="$1" score="$2"
  local -a unset_args=() target_assignments=()
  if [[ "${model_kind}" == "api" && "${score}" != "1" ]]; then
    require_api_key "${api_key_variable}"
  fi
  case "${target_mode}" in
    canary)
      unset_args=(-u INDICES -u MSMU_SMOKE_INDICES)
      target_assignments=(LIMIT=1)
      if [[ "${model_kind}" == "api" ]]; then target_assignments=(LIMIT=2); fi
      ;;
    smoke)
      unset_args=(-u LIMIT)
      target_assignments=(INDICES="${MSMU_SMOKE_INDICES}")
      ;;
    full)
      unset_args=(-u LIMIT -u INDICES -u MSMU_SMOKE_INDICES)
      ;;
    *) fail "internal error: unsupported target mode ${target_mode}" ;;
  esac

  case "${model_kind}" in
    api)
      run_command env "${unset_args[@]}" -u INFERENCE_BASE_URL \
        ${target_assignments[@]+"${target_assignments[@]}"} \
        RUN_NAME="${run_name}" RUN_SCORE="${score}" \
        SCORE_ONLY="${score}" \
        JUDGE_BASE_URL="${judge_base_url}" PROFILE="${profile}" BACKEND="${api_backend}" \
          bash "${SCRIPT_DIR}/run_openai_compatible_pipeline.sh"
      ;;
    vllm)
      run_command env "${unset_args[@]}" ${target_assignments[@]+"${target_assignments[@]}"} \
        RUN_NAME="${run_name}" RUN_SCORE="${score}" \
        SCORE_ONLY="${score}" \
        JUDGE_BASE_URL="${judge_base_url}" PROFILE="${profile}" BACKEND=vllm \
        SERVED_MODEL_NAME="${served_model_name}" INFERENCE_BASE_URL="${inference_base_url}" \
          bash "${SCRIPT_DIR}/run_openai_compatible_pipeline.sh"
      ;;
    qwen|qwen_peft)
      require_value QWEN_MODEL "${qwen_model}"
      require_value QWEN_REVISION "${qwen_revision}"
      if [[ "${score}" != "1" ]]; then qwen_gpu_preflight; fi
      local -a qwen_args=(
        env "${unset_args[@]}" ${target_assignments[@]+"${target_assignments[@]}"}
        RUN_NAME="${run_name}" RUN_SCORE="${score}" SCORE_ONLY="${score}"
        JUDGE_BASE_URL="${judge_base_url}"
        PROFILE="${profile}" BASE_MODEL="${qwen_model}" BASE_MODEL_REVISION="${qwen_revision}"
        CUDA_VISIBLE_DEVICES="${MANUAL_CUDA_VISIBLE_DEVICES:-${default_devices}}"
        DEVICE_MAP="${qwen_device_map}" BATCH_SIZE="${qwen_batch_size}"
      )
      if [[ "${model_kind}" == "qwen_peft" ]]; then
        qwen_args+=(CHECKPOINT="${QWEN_PEFT_CHECKPOINT}" CHECKPOINT_REVISION="${QWEN_PEFT_REVISION:-}")
      fi
      if [[ "${target_mode}" == "canary" ]]; then
        qwen_args+=(QWEN_VISION_CANARY=1)
      fi
      qwen_args+=(bash "${SCRIPT_DIR}/run_qwen_peft_pipeline.sh")
      run_command "${qwen_args[@]}"
      ;;
    ssr)
      run_command env "${unset_args[@]}" ${target_assignments[@]+"${target_assignments[@]}"} \
        RUN_NAME="${run_name}" RUN_SCORE="${score}" SCORE_ONLY="${score}" \
        JUDGE_BASE_URL="${judge_base_url}" PROFILE="${profile}" \
          bash "${SCRIPT_DIR}/run_ssr_pipeline.sh"
      ;;
    spatialrgpt)
      require_value SPATIALRGPT_MODEL "${model_path}"
      run_command env "${unset_args[@]}" ${target_assignments[@]+"${target_assignments[@]}"} \
        RUN_NAME="${run_name}" RUN_SCORE="${score}" SCORE_ONLY="${score}" \
        JUDGE_BASE_URL="${judge_base_url}" MODEL_PATH="${model_path}" \
          bash "${SCRIPT_DIR}/run_spatialrgpt_pipeline.sh"
      ;;
    3dthinker)
      require_value THREEDTHINKER_MODEL "${model_path}"
      run_command env "${unset_args[@]}" ${target_assignments[@]+"${target_assignments[@]}"} \
        RUN_NAME="${run_name}" RUN_SCORE="${score}" SCORE_ONLY="${score}" \
        JUDGE_BASE_URL="${judge_base_url}" PROFILE="${profile}" MODEL_PATH="${model_path}" \
          bash "${SCRIPT_DIR}/run_3dthinker_pipeline.sh"
      ;;
    spatialbot)
      require_value SPATIALBOT_MODEL "${model_path}"
      run_command env "${unset_args[@]}" ${target_assignments[@]+"${target_assignments[@]}"} \
        RUN_NAME="${run_name}" RUN_SCORE="${score}" SCORE_ONLY="${score}" \
        JUDGE_BASE_URL="${judge_base_url}" PROFILE="${profile}" MODEL_PATH="${model_path}" \
          bash "${SCRIPT_DIR}/run_spatialbot_pipeline.sh"
      ;;
    *) fail "internal error: ${model_kind} has no inference pipeline" ;;
  esac
}

select_smoke_indices() {
  local report="${OUTPUT_ROOT}/02_smoke8/selected_indices.json"
  if [[ "${manual_dry_run}" == "1" ]]; then
    run_command env DATASET_ROOT="${DATASET_ROOT}" SMOKE_INDEX_REPORT="${report}" \
      bash "${SCRIPT_DIR}/select_smoke_indices.sh"
    MSMU_SMOKE_INDICES="benchmark-selected-8-indices"
  else
    MSMU_SMOKE_INDICES="$(
      DATASET_ROOT="${DATASET_ROOT}" SMOKE_INDEX_REPORT="${report}" \
        bash "${SCRIPT_DIR}/select_smoke_indices.sh"
    )"
  fi
  export MSMU_SMOKE_INDICES
  printf '[msmu-manual] smoke_indices=%s\n' "${MSMU_SMOKE_INDICES}"
}

serve_judge() {
  require_value JUDGE_MODEL "${JUDGE_MODEL:-}"
  local devices="${MANUAL_JUDGE_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
  local check_dir="${OUTPUT_ROOT}/03_full987/judge"
  run_command env CUDA_VISIBLE_DEVICES="${devices}" \
    MIN_FREE_GPU_MIB="${JUDGE_MIN_FREE_GPU_MIB:-35000}" \
      bash "${SCRIPT_DIR}/gpu_preflight.sh"
  run_logged_command "${check_dir}/vllm_serve.log" \
    env CUDA_VISIBLE_DEVICES="${devices}" JUDGE_MODEL="${JUDGE_MODEL}" \
    JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-msmu-judge}" HOST=127.0.0.1 PORT=18080 \
      bash "${SCRIPT_DIR}/serve_local_judge.sh"
}

case "${stage}" in
  1)
    if [[ "${model_kind}" == "judge" ]]; then fail "judge is not a stage-1 model"; fi
    if [[ "${model}" == "internvl3_78b" ]]; then
      action="${action:-check}"
      if [[ "${action}" != "check" ]]; then
        blocked "internvl3_78b only allows stage-1 static check; serve/canary are blocked"
      fi
      preflight_vllm
      dry_run_internvl78_config
    elif [[ "${model_kind}" == "vllm" ]]; then
      if [[ -z "${action}" ]]; then
        fail "${model} needs an action: serve in terminal A, check in terminal B"
      fi
      case "${action}" in
        serve) preflight_vllm; serve_vllm ;;
        check) check_vllm_canary ;;
        *) fail "stage 1 vLLM action must be serve or check" ;;
      esac
    else
      action="${action:-run}"
      if [[ "${action}" != "run" ]]; then fail "stage 1 action for ${model} must be run"; fi
      run_model_pipeline canary 0
    fi
    ;;
  2)
    if [[ "${model}" == "internvl3_78b" ]]; then
      blocked "internvl3_78b is not approved for stage 2"
    fi
    if [[ "${model_kind}" == "judge" ]]; then fail "judge is not a stage-2 model"; fi
    action="${action:-run}"
    if [[ "${action}" == "serve" ]]; then
      if [[ "${model_kind}" != "vllm" ]]; then fail "only vLLM models use the stage-2 serve action"; fi
      serve_vllm
    elif [[ "${action}" == "run" ]]; then
      select_smoke_indices
      run_model_pipeline smoke 0
    else
      fail "stage 2 action must be serve or run"
    fi
    ;;
  3)
    if [[ "${model_kind}" == "judge" ]]; then
      action="${action:-serve}"
      if [[ "${action}" != "serve" ]]; then fail "judge only supports the serve action"; fi
      serve_judge
      exit 0
    fi
    if [[ "${model}" == "internvl3_78b" || "${model}" == "qwen25_vl_72b" ]]; then
      blocked "${model} is excluded from stage 3 because 70B+ models are outside the accepted test plan"
    fi
    action="${action:-infer}"
    case "${action}" in
      serve)
        if [[ "${model_kind}" != "vllm" ]]; then fail "only vLLM models use the stage-3 serve action"; fi
        serve_vllm
        ;;
      infer)
        run_model_pipeline full 0
        ;;
      score)
        if [[ "${model_kind}" == "vllm" && "${inference_base_url}" == "${judge_base_url}" ]]; then
          fail "judge and tested-model endpoints must be different"
        fi
        run_model_pipeline full 1
        ;;
      *) fail "stage 3 action must be serve, infer, or score" ;;
    esac
    ;;
esac
