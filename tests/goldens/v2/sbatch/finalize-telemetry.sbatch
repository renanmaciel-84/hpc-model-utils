#!/bin/bash
#SBATCH --job-name=study-newave
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=1-00:00:00
#SBATCH --chdir=<WS>
#SBATCH --output=<WS>/.hpcmu/logs/finalize-%j.out
set -u
export PATH=/opt/mpich/bin:/opt/slurm/bin:"$PATH"
export HPCMU_PLATFORM=off
unset PYTHONPATH PYTHONHOME
export PYTHONSAFEPATH=1
echo "HPCMU_START job=${SLURM_JOB_ID:-?} restart=${SLURM_RESTART_COUNT:-0} at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "HPCMU_SHAPE nodes=${SLURM_JOB_NUM_NODES:-?} cpus_on_node=${SLURM_CPUS_ON_NODE:-?} nproc=$(nproc) mem_per_node=${SLURM_MEM_PER_NODE:-?} mem_total_kb=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo 2>/dev/null || echo ?)"
ulimit -u unlimited 2>/dev/null || ulimit -u 65535 2>/dev/null || true
export POLARS_MAX_THREADS=1 RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
srun --overlap --nodes="${SLURM_JOB_NUM_NODES:-?}" --ntasks="${SLURM_JOB_NUM_NODES:-?}" --ntasks-per-node=1 --cpus-per-task=1 --cpu-bind=none --mpi=none --kill-on-bad-exit=0 --input=none --wait=0 --job-name=hpcmu-telemetry --output=<WS>/.hpcmu/telemetry/finalize-%j-r"${SLURM_RESTART_COUNT:-0}"-%t.log /opt/hpcmu/python -I -m hpc_model_utils.infra.telemetry.sampler --staging <WS>/.hpcmu/telemetry --phase finalize --workspace <WS> --interval 5 --workspace-interval 60 --tool-version 9.9.9 > <WS>/.hpcmu/telemetry/finalize-"${SLURM_JOB_ID:-?}"-r"${SLURM_RESTART_COUNT:-0}"-srun.log 2>&1 & disown
exec /opt/hpcmu/bin finalize newave --model-job-id 6208 --cores 64 --synthesis-bin '/shared-zfs/tools/sintetizador-newave/<sha>/.venv/bin/sintetizador-newave'
