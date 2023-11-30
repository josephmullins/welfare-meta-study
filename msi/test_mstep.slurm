#!/bin/bash -l        
#SBATCH --time=1:00:00
#SBATCH --ntasks=80
#SBATCH --mem=140g
#SBATCH --tmp=140g
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=mullinsj@umn.edu 
#SBATCH --output=output/TestMStep.out

cd ~/welfare-meta-study
~/julia-1.9.3/bin/julia -t 80 tests/test_mstep_K5.jl
