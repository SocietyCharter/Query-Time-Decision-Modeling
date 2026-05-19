#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${ROOT_DIR}/arbiter-service.env" ]]; then
  set -a
  source "${ROOT_DIR}/arbiter-service.env"
  set +a
fi

exec uvicorn qtdm_arbiter.api:create_app --factory --host 0.0.0.0 --port "${QTDM_PORT:-8001}"
