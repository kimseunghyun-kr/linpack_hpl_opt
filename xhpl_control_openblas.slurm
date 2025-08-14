#!/bin/bash
#SBATCH --job-name=xhpl_single_control
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --constraint=xcnf
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --chdir=/home/k/kimsh/HPL_Linpack/hpl-2.3/bin/EPYC_openblas
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --export=NONE

set -euo pipefail

RUNROOT="$HOME/HPL_Linpack/hpl-2.3/bin/EPYC_openblas"
mkdir -p "$RUNROOT/results" "$RUNROOT/logs"

HPL_DAT_SRC="${1:-}"; TAG="${2:-}"
export PATH="$HOME/.local/mpich/bin:/usr/bin:/bin:$PATH"
OPENBLAS_HOME="$HOME/opt/openblas"
export LD_LIBRARY_PATH="$OPENBLAS_HOME/lib:$HOME/.local/mpich/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENBLAS_VERBOSE=2
export OPENBLAS_CORETYPE=Zen
EXPORTS="PATH,LD_LIBRARY_PATH,OMP_NUM_THREADS,OPENBLAS_NUM_THREADS,MKL_NUM_THREADS,OPENBLAS_VERBOSE,OPENBLAS_CORETYPE"

[[ -n "${HPL_DAT_SRC:-}" ]] || { echo "[xhpl_single_control] ERROR: HPL_DAT_SRC not provided" >&2; exit 2; }
[[ -n "$TAG" ]] || TAG="job${SLURM_JOB_ID}"

OUT="$RUNROOT/results/xhpl_${TAG}.out"
STDLOG="$RUNROOT/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.stdout"

# Stage per-job work dir and run with a file literally named HPL.dat
WORK="$RUNROOT/work/$TAG"
mkdir -p "$WORK"
cp "$HPL_DAT_SRC" "$WORK/HPL.dat"
ln -sf "$RUNROOT/xhpl" "$WORK/xhpl"

echo "[xhpl_single_control] CWD: $WORK"
echo "[xhpl_single_control] DAT line3: $(awk 'NR==3{print; exit}' "$WORK/HPL.dat")"
echo "[xhpl_single_control] OUT expected: $OUT"

# Sanity print on this node: confirm OpenBLAS is linked
srun -N1 -n1 --mpi=pmi2 --chdir="$WORK" --export=PATH,LD_LIBRARY_PATH \
  /bin/bash -lc 'echo -n "$(hostname -s)  "; ldd ./xhpl | egrep -i "openblas|blis|mpich|mpi" | head -n3' || true

# Run HPL (BLAS single-threaded; place ranks densely; keep memory local; try to hold high CPU freq)
srun --nodes=1 --ntasks=64 --mpi=pmi2 \
     --cpu-bind=cores --hint=nomultithread \
     --distribution=block:block --mem-bind=local --cpu-freq=High \
     --chdir="$WORK" --export=$EXPORTS \
     ./xhpl | tee "$STDLOG"

ls -l "$OUT" || echo "[xhpl_single_control] Result file not found at $OUT"

