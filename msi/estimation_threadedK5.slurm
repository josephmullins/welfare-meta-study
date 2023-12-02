#!/bin/bash -l        
#SBATCH --time=2:00:00
#SBATCH --ntasks=80
#SBATCH --mem=100g
#SBATCH --tmp=100g
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=mullinsj@umn.edu 
#SBATCH --output=output/EstimationK5.out

cd ~/welfare-meta-study
~/julia-1.9.3/bin/julia -t 80 scripts/estimate_model_childsample_K5.jl
