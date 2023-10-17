#!/bin/bash -l        
#SBATCH --time=24:00:00
#SBATCH --ntasks=120
#SBATCH --mem=100g
#SBATCH --tmp=100g
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=mullinsj@umn.edu 
#SBATCH --output=Estimation.out

cd ~/welfare-meta-study
~/julia-1.9.3/bin/julia -t 120 scripts/estimate_model_childsample.jl
