#!/bin/bash -l        
#SBATCH --time=5:00:00
#SBATCH --ntasks=50
#SBATCH --mem=100g
#SBATCH --tmp=100g
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=mullinsj@umn.edu 
#SBATCH --output=output/EstimationK6.out

cd ~/welfare-meta-study
~/julia-1.9.3/bin/julia -t 50 scripts/estimate_model_childsample_K6.jl
