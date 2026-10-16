readonly HPCMU_EXECUTION_ID='{{CurrentExecution.ExecutionId}}'
set -euo pipefail
fail() { printf 'hpcmu-task: %s\n' "$1" >&2; exit 2; }
[[ "$HPCMU_EXECUTION_ID" =~ ^[A-Za-z0-9-]{1,64}$ ]] || fail 'invalid CurrentExecution.ExecutionId'

IFS= read -r -d '' MODEL <<'HPCMU_{{CurrentExecution.ExecutionId}}' || :
{{modelName}}
HPCMU_{{CurrentExecution.ExecutionId}}
MODEL=${MODEL%$'\n'}
[[ "$MODEL" =~ ^cobre$ ]] || fail 'invalid modelName'

IFS= read -r -d '' ROOT_PATH <<'HPCMU_{{CurrentExecution.ExecutionId}}' || :
{{rootPath}}
HPCMU_{{CurrentExecution.ExecutionId}}
ROOT_PATH=${ROOT_PATH%$'\n'}
[[ "$ROOT_PATH" == '@@env:rootPath@@' ]] || fail 'invalid rootPath'

IFS= read -r -d '' WORKDIR <<'HPCMU_{{CurrentExecution.ExecutionId}}' || :
{{path}}
HPCMU_{{CurrentExecution.ExecutionId}}
WORKDIR=${WORKDIR%$'\n'}
[[ "$WORKDIR" =~ ^/[A-Za-z0-9._/-]+/cobre_[A-Za-z0-9]{6}$ ]] || fail 'invalid path'
[[ "${WORKDIR%/*}" == "${ROOT_PATH%/}" ]] || fail 'invalid path'
[[ "${WORKDIR##*/}" == "${MODEL}_"* ]] || fail 'invalid path'

IFS= read -r -d '' QUEUE <<'HPCMU_{{CurrentExecution.ExecutionId}}' || :
{{queue}}
HPCMU_{{CurrentExecution.ExecutionId}}
QUEUE=${QUEUE%$'\n'}
[[ "$QUEUE" =~ ^[A-Za-z0-9_-]{1,64}$ ]] || fail 'invalid queue'

IFS= read -r -d '' CORES <<'HPCMU_{{CurrentExecution.ExecutionId}}' || :
{{coreCount}}
HPCMU_{{CurrentExecution.ExecutionId}}
CORES=${CORES%$'\n'}
[[ "$CORES" =~ ^[1-9][0-9]{0,5}$ ]] || fail 'invalid coreCount'

IFS= read -r -d '' MAX_CORES <<'HPCMU_{{CurrentExecution.ExecutionId}}' || :
{{maxCoresPerNode}}
HPCMU_{{CurrentExecution.ExecutionId}}
MAX_CORES=${MAX_CORES%$'\n'}
[[ "$MAX_CORES" =~ ^[1-9][0-9]{0,3}$ ]] || fail 'invalid maxCoresPerNode'

IFS= read -r -d '' JOB_HOURS <<'HPCMU_{{CurrentExecution.ExecutionId}}' || :
{{jobTimeoutHours}}
HPCMU_{{CurrentExecution.ExecutionId}}
JOB_HOURS=${JOB_HOURS%$'\n'}
[[ "$JOB_HOURS" =~ ^[1-9][0-9]{0,3}$ ]] || fail 'invalid jobTimeoutHours'

IFS= read -r -d '' MPICH_PATH <<'HPCMU_{{CurrentExecution.ExecutionId}}' || :
{{mpichPath}}
HPCMU_{{CurrentExecution.ExecutionId}}
MPICH_PATH=${MPICH_PATH%$'\n'}
[[ "$MPICH_PATH" == '@@env:cobreMpichPath@@' ]] || fail 'invalid mpichPath'

IFS= read -r -d '' SLURM_PATH <<'HPCMU_{{CurrentExecution.ExecutionId}}' || :
{{slurmPath}}
HPCMU_{{CurrentExecution.ExecutionId}}
SLURM_PATH=${SLURM_PATH%$'\n'}
[[ "$SLURM_PATH" == '@@env:slurmPath@@' ]] || fail 'invalid slurmPath'

IFS= read -r -d '' UTILS_DIR <<'HPCMU_{{CurrentExecution.ExecutionId}}' || :
{{utilsToolDir}}
HPCMU_{{CurrentExecution.ExecutionId}}
UTILS_DIR=${UTILS_DIR%$'\n'}
[[ "$UTILS_DIR" =~ ^/[A-Za-z0-9._/-]+/hpc-model-utils/[0-9a-f]{40}$ ]] || fail 'invalid utilsToolDir'
[[ "${UTILS_DIR%/hpc-model-utils/*}" == '@@env:toolsRoot@@' ]] || fail 'invalid utilsToolDir'

IFS= read -r -d '' SYNTHESIS_DIR <<'HPCMU_{{CurrentExecution.ExecutionId}}' || :
{{synthesisToolDir}}
HPCMU_{{CurrentExecution.ExecutionId}}
SYNTHESIS_DIR=${SYNTHESIS_DIR%$'\n'}
[[ "$SYNTHESIS_DIR" =~ ^/[A-Za-z0-9._/-]+/novomodelo-bridge/[0-9a-f]{40}$ ]] || fail 'invalid synthesisToolDir'
[[ "${SYNTHESIS_DIR%/novomodelo-bridge/*}" == '@@env:toolsRoot@@' ]] || fail 'invalid synthesisToolDir'

cd -- "$WORKDIR"
exec "$UTILS_DIR/.venv/bin/hpc-model-utils" run "$MODEL" "$QUEUE" "$CORES" --max-cores-per-node "$MAX_CORES" --max-job-time-hours "$JOB_HOURS" --mpich-path "$MPICH_PATH" --slurm-path "$SLURM_PATH" --synthesis-bin "$SYNTHESIS_DIR/.venv/bin/novomodelo-bridge"
