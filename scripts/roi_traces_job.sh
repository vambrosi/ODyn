#!/bin/bash
#SBATCH --partition=batch
#SBATCH --job-name=roi_traces
#SBATCH --output="../logs/roi_traces_%j.out"
#SBATCH --error="../logs/roi_traces_%j.out"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=8:00:00

# Save the ROI traces of several groups.
#
#   sbatch roi_traces_job.sh --project PROJECT_NAME --groups 48 49 53
#
# Arguments are all passed to save_roi_traces.py, so see its --help for
# reference. A couple of CPUs only because the work is mostly loading files.
# Memory requirements is one movie window plus the traces.

set -euo pipefail

export PYTHONUNBUFFERED=1
export TQDM_DISABLE=1

source "$HOME/miniforge3/etc/profile.d/conda.sh"
set +u
conda activate caiman
set -u

cd "$(dirname "$0")"

echo "STARTING ROI TRACES"
echo "python   $(which python)"
echo "commit   $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "job      ${SLURM_JOB_ID:-none} on $(hostname)"
echo "started  $(date --iso-8601=seconds)"
echo "args     $*"
echo

python save_roi_traces.py "$@"

echo
echo "finished $(date --iso-8601=seconds)"
