#!/bin/bash -l        
#SBATCH --time=6:00:00
#SBATCH --ntasks=50
#SBATCH --mem=90g
#SBATCH --tmp=90g
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=mullinsj@umn.edu 
#SBATCH --output=output/EstimationK4.out

cd ~/welfare-meta-study
~/julia-1.9.3/bin/julia -t 50 scripts/estimate_model_childsample_K4.jl
