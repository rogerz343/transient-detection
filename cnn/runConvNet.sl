#!/bin/bash

#SBATCH -q regular
#SBATCH -N 2
#SBATCH -t 2:00:00
#SBATCH -L SCRATCH
#SBATCH -C haswell

cd ${SLURM_SUBMIT_DIR}
sh runConvNet.sh 
