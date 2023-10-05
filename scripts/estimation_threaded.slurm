#!/bin/bash -l        
#SBATCH --time=36:00:00
#SBATCH --ntasks=120
#SBATCH --mem=120g
#SBATCH --tmp=120g
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=mullinsj@umn.edu 
#SBATCH --output=Estimation.out

cd ~/welfare-meta-study
~/julia-1.9.3/bin/julia -t 120 scripts/estimate_model_childsample.jl
