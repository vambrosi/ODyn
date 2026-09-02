#!/bin/bash
#SBATCH --partition=batch
#SBATCH --job-name=scatter
#SBATCH --output="../logs/scatter_%j.out"
#SBATCH --error="../logs/scatter_%j.out"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=8:00:00

# Compare programs pixel by pixel, for one or more groups.
#
#   sbatch scatter_job.sh --project PA_K99 --groups 48 49 53
#   sbatch scatter_job.sh --project PA_K99 --groups 48 --outcomes all
#
# Arguments are all passed to program_scatter.py, so see its --help
# for reference.

set -euo pipefail

export PYTHONUNBUFFERED=1
export TQDM_DISABLE=1
export MPLBACKEND=Agg

source "$HOME/miniforge3/etc/profile.d/conda.sh"
set +u
conda activate caiman
set -u

cd "$(dirname "$0")"

echo "STARTING PROGRAM SCATTER"
echo "python   $(which python)"
echo "commit   $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "job      ${SLURM_JOB_ID:-none} on $(hostname)"
echo "cores    ${SLURM_CPUS_PER_TASK:-?}"
echo "started  $(date --iso-8601=seconds)"
echo "args     $*"
echo

python program_scatter.py "$@"

echo
echo "finished $(date --iso-8601=seconds)"
