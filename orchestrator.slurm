#!/bin/bash
#SBATCH -J orchestrator
#SBATCH --mem=1G
#SBATCH -t 1-00:00:00
#SBATCH -o orchestrator.%j.log
#SBATCH --export=NONE

source ~/.bashrc
python3 -u ./hpl_sweeper_openblas.py

