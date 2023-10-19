#!/bin/bash -l        
#SBATCH --time=12:00:00
#SBATCH --ntasks=70
#SBATCH --mem=120g
#SBATCH --tmp=120g
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=mullinsj@umn.edu 
#SBATCH --output=output/Estimation.out

cd ~/welfare-meta-study
~/julia-1.9.3/bin/julia -t 70 scripts/estimate_model_childsample.jl
