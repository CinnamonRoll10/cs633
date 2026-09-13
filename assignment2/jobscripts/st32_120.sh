#!/bin/bash
#SBATCH --job-name=st32_120
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --time=00:08:00
#SBATCH --partition=standard

module load compiler/openmpi/4.1.5

mpirun -np 32 ./stencil 7 32 4 4 2 120 120 120 5 1000 2 500