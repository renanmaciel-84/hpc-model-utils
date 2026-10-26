readonly HPCMU_EXECUTION_ID='{{CurrentExecution.ExecutionId}}'
set -euo pipefail
fail() { printf 'hpcmu-task: %s\n' "$1" >&2; exit 2; }
[[ "$HPCMU_EXECUTION_ID" =~ ^[A-Za-z0-9-]{1,64}$ ]] || fail 'invalid CurrentExecution.ExecutionId'

IFS= read -r -d '' UTILS_TAG <<'HPCMU_{{CurrentExecution.ExecutionId}}' || :
{{utilsAppVersion}}
HPCMU_{{CurrentExecution.ExecutionId}}
UTILS_TAG=${UTILS_TAG%$'\n'}
[[ "$UTILS_TAG" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] || fail 'invalid utilsAppVersion'

IFS= read -r -d '' UTILS_SHA <<'HPCMU_{{CurrentExecution.ExecutionId}}' || :
{{utilsAppSha}}
HPCMU_{{CurrentExecution.ExecutionId}}
UTILS_SHA=${UTILS_SHA%$'\n'}
[[ "$UTILS_SHA" =~ ^[0-9a-f]{40}$ ]] || fail 'invalid utilsAppSha'

IFS= read -r -d '' MODEL_VERSION <<'HPCMU_{{CurrentExecution.ExecutionId}}' || :
{{modelVersion}}
HPCMU_{{CurrentExecution.ExecutionId}}
MODEL_VERSION=${MODEL_VERSION%$'\n'}
[[ "$MODEL_VERSION" =~ ^v[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*)?$ ]] || fail 'invalid modelVersion'

# A patched build (v0.19.1-fixinflows) runs the bridge of its base release.
BRIDGE_TAG=${MODEL_VERSION%%-*}

exec bash -s -- --root '@@env:toolsRoot@@' --uv '@@env:uvBin@@' --python '3.12.13' --tool hpc-model-utils https://github.com/rjmalves/hpc-model-utils.git "$UTILS_TAG" "$UTILS_SHA" hpc-model-utils --tool-tag novomodelo-bridge https://github.com/ons-ccee-epe/novomodelo-bridge.git "$BRIDGE_TAG" novomodelo-bridge <<'HPCMU_ENSURE_TOOLS_EOF'
@@script:ensure-tools.sh@@
HPCMU_ENSURE_TOOLS_EOF
