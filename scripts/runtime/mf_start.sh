#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

cd "${REPO_ROOT}"

TARGET_YAML="${1:-${REPO_ROOT}/assets/runtime/mf_templates/qwen2.yaml}"
CARD_NUM="${2:-8}"

# MindFormers trainer init requires hostname -> IP resolution.
# In some containers /etc/hosts lacks the current hostname, causing socket.gaierror.
CURRENT_HOSTNAME="$(hostname 2>/dev/null || true)"
if [[ -n "${CURRENT_HOSTNAME}" ]] && ! getent hosts "${CURRENT_HOSTNAME}" >/dev/null 2>&1; then
	if [[ -w /etc/hosts ]]; then
		echo "127.0.0.1 ${CURRENT_HOSTNAME}" >> /etc/hosts
		echo "[LMSV][mf_start] Added hostname mapping to /etc/hosts: ${CURRENT_HOSTNAME} -> 127.0.0.1"
	else
		echo "[LMSV][mf_start][WARN] Hostname is not resolvable and /etc/hosts is not writable: ${CURRENT_HOSTNAME}" >&2
	fi
fi

printf -v EXECUTE_ORDER 'python %q --config %q' "utils/runtime/run_mindformer.py" "${TARGET_YAML}"
if [[ -n "${LMSV_MF_WORKER_NUM:-}" && -n "${LMSV_MF_LOCAL_WORKER:-}" && -n "${LMSV_MF_MASTER_ADDR:-}" && -n "${LMSV_MF_NODE_RANK:-}" ]]; then
  MF_LOG_DIR="${LMSV_MF_LOG_DIR:-msrun_log}"
  MF_JOIN="${LMSV_MF_JOIN:-True}"
  MF_CLUSTER_TIMEOUT="${LMSV_MF_CLUSTER_TIMEOUT:-300}"
  MF_MASTER_PORT="${LMSV_MF_MASTER_PORT:-${LMSV_MSRUN_MASTER_PORT:-8118}}"
  bash "${REPO_ROOT}/scripts/runtime/msrun_launcher.sh" \
    "${EXECUTE_ORDER}" \
    "${LMSV_MF_WORKER_NUM}" \
    "${LMSV_MF_LOCAL_WORKER}" \
    "${LMSV_MF_MASTER_ADDR}" \
    "${MF_MASTER_PORT}" \
    "${LMSV_MF_NODE_RANK}" \
    "${MF_LOG_DIR}" \
    "${MF_JOIN}" \
    "${MF_CLUSTER_TIMEOUT}"
else
  bash "${REPO_ROOT}/scripts/runtime/msrun_launcher.sh" "${EXECUTE_ORDER}" "${CARD_NUM}"
fi
