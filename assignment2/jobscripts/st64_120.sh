#!/bin/bash
#SBATCH --job-name=st64_120
#SBATCH --nodes=2
#SBATCH --ntasks=64
#SBATCH --time=00:08:00
#SBATCH --partition=standard

module load compiler/openmpi/4.1.5

mpirun -np 64 ./stencil 7 32 4 4 4 120 120 120 5 1000 2 500