#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "${PROFILE:-}" in internvl3_8b|internvl3_38b|internvl3_78b) ;; *) echo "Set an InternVL3 PROFILE" >&2; exit 2 ;; esac
exec "${SCRIPT_DIR}/serve_vllm_profile.sh"
