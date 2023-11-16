#!/bin/bash -l        
#SBATCH --time=4:00:00
#SBATCH --ntasks=20
#SBATCH --mem=150g
#SBATCH --tmp=150g
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=mullinsj@umn.edu 
#SBATCH --output=output/Counterfactuals.out

cd ~/welfare-meta-study
~/julia-1.9.3/bin/julia -t 20 scripts/decomposition_counterfactual.jl

