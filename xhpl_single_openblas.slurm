#!/bin/bash
#SBATCH --job-name=xhpl_single
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=64
# (remove or set to partition value) #SBATCH --cpus-per-task=16
#SBATCH --constraint=xcnf
#SBATCH --mem=900G
#SBATCH --time=01:00:00
#SBATCH --chdir=/home/k/kimsh/HPL_Linpack/hpl-2.3/bin/EPYC_openblas
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --export=NONE

set -euo pipefail

RUNROOT="$HOME/HPL_Linpack/hpl-2.3/bin/EPYC_openblas"
mkdir -p "$RUNROOT/results" "$RUNROOT/logs"

HPL_DAT_SRC="${1:-}"; TAG="${2:-}"
[[ -n "${HPL_DAT_SRC:-}" ]] || { echo "[xhpl_single] ERROR: HPL_DAT_SRC not provided" >&2; exit 2; }
[[ -n "$TAG" ]] || TAG="job${SLURM_JOB_ID}"

# Toolchain + BLAS (make sure OpenBLAS is found everywhere)
export PATH="$HOME/.local/mpich/bin:/usr/bin:/bin:$PATH"
OPENBLAS_HOME="$HOME/opt/openblas"
export LD_LIBRARY_PATH="$OPENBLAS_HOME/lib:$HOME/.local/mpich/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENBLAS_CORETYPE=Zen
export OPENBLAS_VERBOSE=1

# PSM3 provider, pin to your NIC/subnet, **quiet by default**
export FI_PROVIDER=psm3
export PSM3_NIC=enp131s0f0
export PSM3_ADDR_FMT=4
export PSM3_SUBNETS=192.168.48.0/21
export PSM3_ALLOW_ROUTERS=0
export PSM3_TCP_BIND_SRC=1
export PSM3_IDENTIFY=0  # <- set to 1 to print those lines again

EXPORTS="PATH,LD_LIBRARY_PATH,OMP_NUM_THREADS,OPENBLAS_NUM_THREADS,MKL_NUM_THREADS,OPENBLAS_CORETYPE,OPENBLAS_VERBOSE,FI_PROVIDER,PSM3_NIC,PSM3_ADDR_FMT,PSM3_SUBNETS,PSM3_ALLOW_ROUTERS,PSM3_TCP_BIND_SRC,PSM3_IDENTIFY"

OUT="$RUNROOT/results/xhpl_${TAG}.out"
STDLOG="$RUNROOT/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.stdout"

# Stage per-job work dir
WORK="$RUNROOT/work/$TAG"
mkdir -p "$WORK"
cp "$HPL_DAT_SRC" "$WORK/HPL.dat"
ln -sf "$RUNROOT/xhpl" "$WORK/xhpl"

echo "[xhpl_single] CWD: $WORK"
echo "[xhpl_single] DAT line3: $(awk 'NR==3{print; exit}' "$WORK/HPL.dat")"
echo "[xhpl_single] OUT expected: $OUT"

# Quick per-node sanity: show BLAS lib and the IPv4 bound to the NIC
srun -N4 -n4 --mpi=pmi2 --chdir="$WORK" --export=PATH,LD_LIBRARY_PATH \
  /bin/bash -lc 'printf "%-7s " "$(hostname -s)"; ldd ./xhpl | egrep -i "openblas|mpich|mpi" | head -n1 | sed "s/^[[:space:]]*//"; printf " | "; ip -4 -o addr show dev enp131s0f0 | awk "{print \$4}"' || true

# Run HPL (quiet PSM3)
srun --nodes=4 --ntasks-per-node=64 --mpi=pmi2 \
     --cpu-bind=cores --hint=nomultithread \
     --distribution=block:block --mem-bind=local \
     --chdir="$WORK" --export=$EXPORTS \
     ./xhpl | tee "$STDLOG"

ls -l "$OUT" || echo "[xhpl_single] Result file not found at $OUT"

