#!/bin/bash -l        
#SBATCH --time=12:00:00
#SBATCH --ntasks=120
#SBATCH --mem=120g
#SBATCH --tmp=120g
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=mullinsj@umn.edu 
#SBATCH --output=CheckEstimates.out

cd ~/welfare-meta-study
~/julia-1.9.3/bin/julia -t 120 scripts/check_estimates.jl
