#!/bin/bash -l
#
#SBATCH --job-name="errormc"
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --partition=batch
#SBATCH --wait

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CRAY_CUDA_MPS=1
source ~/error-mc/numpy_model/MCenv/activate.sh
#pwd
#module list
which python

echo Starting
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
$@
