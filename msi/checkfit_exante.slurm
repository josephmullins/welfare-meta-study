#!/bin/bash -l        
#SBATCH --time=00:05:00
#SBATCH --ntasks=20
#SBATCH --mem=100g
#SBATCH --tmp=100g
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=mullinsj@umn.edu 
#SBATCH --output=output/checkfit.out

cd ~/welfare-meta-study
~/julia-1.9.3/bin/julia -t 20 scripts/checkfit_exante.jl
