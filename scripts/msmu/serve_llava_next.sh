#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "${PROFILE:-}" in llava_next_mistral_7b|llava_next_yi_34b) ;; *) echo "Set a LLaVA-NeXT PROFILE" >&2; exit 2 ;; esac
exec "${SCRIPT_DIR}/serve_vllm_profile.sh"
