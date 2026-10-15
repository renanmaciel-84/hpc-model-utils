#!/bin/bash
#SBATCH --job-name=study-decomp
#SBATCH --partition=batch
#SBATCH --exclusive
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --chdir=<WS>
#SBATCH --output=<WS>/.hpcmu/logs/model-%j.out
set -u
export PATH=/opt/mpich/bin:/opt/slurm/bin:"$PATH"
export HPCMU_PLATFORM=off
unset PYTHONPATH PYTHONHOME
export PYTHONSAFEPATH=1
echo "HPCMU_START job=${SLURM_JOB_ID:-?} restart=${SLURM_RESTART_COUNT:-0} at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "HPCMU_SHAPE nodes=${SLURM_JOB_NUM_NODES:-?} cpus_on_node=${SLURM_CPUS_ON_NODE:-?} nproc=$(nproc) mem_per_node=${SLURM_MEM_PER_NODE:-?} mem_total_kb=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo 2>/dev/null || echo ?)"
rm -f <WS>/.hpcmu/jobs/model.exit
srun --overlap --nodes="${SLURM_JOB_NUM_NODES:-?}" --ntasks="${SLURM_JOB_NUM_NODES:-?}" --ntasks-per-node=1 --cpus-per-task=1 --cpu-bind=none --mpi=none --kill-on-bad-exit=0 --input=none --wait=0 --job-name=hpcmu-telemetry --output=<WS>/.hpcmu/telemetry/model-%j-r"${SLURM_RESTART_COUNT:-0}"-%t.log /opt/hpcmu/python -I -m hpc_model_utils.infra.telemetry.sampler --staging <WS>/.hpcmu/telemetry --phase model --workspace <WS> --interval 5 --workspace-interval 60 --tool-version 9.9.9 > <WS>/.hpcmu/telemetry/model-"${SLURM_JOB_ID:-?}"-r"${SLURM_RESTART_COUNT:-0}"-srun.log 2>&1 & disown
mpiexec -np "$SLURM_NTASKS" <WS>/assets/decomp
rc=$?
printf '%s\n' "$rc" > <WS>/.hpcmu/jobs/model.exit
exit "$rc"
