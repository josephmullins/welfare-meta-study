#!/bin/bash -l        
#SBATCH --time=5:00:00
#SBATCH --ntasks=100
#SBATCH --mem=150g
#SBATCH --tmp=150g
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=mullinsj@umn.edu 
#SBATCH --output=output/EstimationK10.out

cd ~/welfare-meta-study
~/julia-1.9.3/bin/julia -t 100 scripts/estimate_model_childsample_K10.jl
