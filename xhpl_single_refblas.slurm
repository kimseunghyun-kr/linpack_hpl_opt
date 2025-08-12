#!/bin/bash
#SBATCH --job-name=xhpl_single
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=64
# (remove or set to partition value) #SBATCH --cpus-per-task=16
#SBATCH --constraint=xcnf
#SBATCH --mem=900G
#SBATCH --time=01:00:00
#SBATCH --chdir=/home/k/kimsh/HPL_Linpack/hpl-2.3/bin/EPYC_refblas
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --export=NONE

set -euo pipefail

RUNROOT="$HOME/HPL_Linpack/hpl-2.3/bin/EPYC_refblas"
mkdir -p "$RUNROOT/results" "$RUNROOT/logs"

HPL_DAT_SRC="${1:-}"; TAG="${2:-}"
export PATH="$HOME/.local/mpich/bin:/usr/bin:/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/mpich/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
FI_BLOCK="FI_PROVIDER=tcp,FI_TCP_IFACE=enp131s0f0"
EXPORTS="PATH,LD_LIBRARY_PATH,OMP_NUM_THREADS,OPENBLAS_NUM_THREADS,MKL_NUM_THREADS,$FI_BLOCK"

[[ -n "${HPL_DAT_SRC:-}" ]] || { echo "[xhpl_single] ERROR: HPL_DAT_SRC not provided" >&2; exit 2; }
[[ -n "$TAG" ]] || TAG="job${SLURM_JOB_ID}"

OUT="$RUNROOT/results/xhpl_${TAG}.out"
STDLOG="$RUNROOT/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.stdout"

# Stage per-job work dir and run with a file literally named HPL.dat
WORK="$RUNROOT/work/$TAG"
mkdir -p "$WORK"
cp "$HPL_DAT_SRC" "$WORK/HPL.dat"
ln -sf "$RUNROOT/xhpl" "$WORK/xhpl"

echo "[xhpl_single] CWD: $WORK"
echo "[xhpl_single] DAT line3: $(awk 'NR==3{print; exit}' "$WORK/HPL.dat")"
echo "[xhpl_single] OUT expected: $OUT"

# sanity print per node
srun -N4 -n4 --mpi=pmi2 --export=PATH,LD_LIBRARY_PATH \
  /bin/bash -lc 'echo -n "$(hostname -s)  "; ldd ./xhpl | egrep -i "libmpich|libmpi" | head -n1' || true

# Run: HPL writes to the path on line 3; we tee stdout to a separate debug log
srun --chdir="$WORK" --mpi=pmi2 --cpu-bind=cores --hint=nomultithread --export=$EXPORTS \
     ./xhpl | tee "$STDLOG"

ls -l "$OUT" || echo "[xhpl_single] Result file not found at $OUT"
