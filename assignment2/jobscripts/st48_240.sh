#!/bin/bash
#SBATCH --job-name=st48_240
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --time=00:08:00
#SBATCH --partition=standard

module load compiler/openmpi/4.1.5

mpirun -np 48 ./stencil 7 48 6 4 2 240 240 240 5 1000 2 500