#!/bin/bash -l        
#SBATCH --time=6:00:00
#SBATCH --ntasks=50
#SBATCH --mem=100g
#SBATCH --tmp=100g
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=mullinsj@umn.edu 
#SBATCH --output=output/Counterfactuals.out

cd $OLDHOME/welfare-meta-study
../julia-1.9.3/bin/julia -t 50 scripts/decomposition_counterfactual.jl
../julia-1.9.3/bin/julia -t 50 scripts/nonselected_counterfactual.jl
