#!/bin/bash -l        
#SBATCH --time=5:00:00
#SBATCH --ntasks=50
#SBATCH --mem=150g
#SBATCH --tmp=150g
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=mullinsj@umn.edu 
#SBATCH --output=output/Estimation2.out

cd ~/welfare-meta-study
~/julia-1.9.3/bin/julia -t 50 scripts/estimate_model_noSIPPv2.jl
