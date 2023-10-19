#!/bin/bash -l        
#SBATCH --time=24:00:00
#SBATCH --ntasks=60
#SBATCH --mem=140g
#SBATCH --tmp=140g
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=mullinsj@umn.edu 
#SBATCH --output=output/EstimationK4.out

cd ~/welfare-meta-study
~/julia-1.9.3/bin/julia -t 60 scripts/estimate_model_childsample_K4.jl
