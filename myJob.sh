#!/bin/bash

#SBATCH --job-name=jobPA
#SBATCH --partition=std
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

## Load software modules


## Move data (if necessary) and launch application
## (use 'srun' for launching mpi applications)

module load python/3.9.19
#module load python/3.13.7

source venv/bin/activate

python test2.py

## Finish gracefully
exit 0
