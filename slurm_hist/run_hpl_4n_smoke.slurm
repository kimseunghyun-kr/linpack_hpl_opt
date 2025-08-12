#!/bin/bash
#SBATCH --job-name=hpl_4r_srun
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --constraint=xcnf
#SBATCH --output=hpl_4r_srun.out
#SBATCH --export=NONE

export PATH=$HOME/.local/mpich/bin:$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd ~/HPL_Linpack/hpl-2.3/bin/EPYC_refblas

srun --mpi=pmi2 --cpu-bind=cores ./xhpl

