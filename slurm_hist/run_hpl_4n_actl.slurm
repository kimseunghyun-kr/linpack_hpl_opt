#!/bin/bash
#SBATCH --job-name=hpl_4nodes
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --constraint=xcnf
#SBATCH --mem=900G
#SBATCH --time=00:30:00
#SBATCH --output=hpl_4nodes.out
#SBATCH --export=NONE   # start from a clean env; we'll whitelist explicitly

# ---- toolchain ----
export PATH="$HOME/.local/mpich/bin:/usr/bin:/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/mpich/lib:${LD_LIBRARY_PATH:-}"

# ---- threading ----
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ---- choose fabric (uncomment ONE block) ----
# PSM3 (fast; mixed subnets → allow routes)
# FI_BLOCK="FI_PROVIDER=psm3,FI_PSM3_IFACE=enp131s0f0,FI_PSM3_ALLOW_ROUTES=1"
# TCP (fallback)
FI_BLOCK="FI_PROVIDER=tcp,FI_TCP_IFACE=enp131s0f0"

# one place to maintain the export list
COMMON_EXPORT="PATH,LD_LIBRARY_PATH,OMP_NUM_THREADS,OPENBLAS_NUM_THREADS,MKL_NUM_THREADS"
ALL_EXPORT="$COMMON_EXPORT,$FI_BLOCK"

cd "$HOME/HPL_Linpack/hpl-2.3/bin/EPYC_refblas"

# (optional) sanity: show MPI lib seen by ranks
srun -N4 -n4 --mpi=pmi2 --export=$COMMON_EXPORT \
  /bin/bash -lc 'echo -n "$(hostname -s)  "; ldd ./xhpl | egrep -i "libmpich|libmpi.so.40" | head -n1'

# run HPL
srun --mpi=pmi2 --cpu-bind=cores --hint=nomultithread --export=$ALL_EXPORT \
  ./xhpl
