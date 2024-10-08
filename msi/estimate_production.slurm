#!/bin/bash -l        
#SBATCH --time=12:00:00
#SBATCH --ntasks=10
#SBATCH --mem=10g
#SBATCH --tmp=10g
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=mullinsj@umn.edu 
#SBATCH --output=output/ProductionEstimation.out

cd $OLDHOME/welfare-meta-study
../julia-1.9.3/bin/julia -t 10 scripts/estimate_production.jl
