#!/bin/bash
#SBATCH --job-name=st96_120
#SBATCH --nodes=2
#SBATCH --ntasks=96
#SBATCH --time=00:08:00
#SBATCH --partition=standard

module load compiler/openmpi/4.1.5

mpirun -np 96 ./stencil 7 48 6 4 4 120 120 120 5 1000 2 500