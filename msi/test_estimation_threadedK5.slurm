#!/bin/bash -l        
#SBATCH --time=0:30:00
#SBATCH --ntasks=50
#SBATCH --mem=100g
#SBATCH --tmp=100g
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=mullinsj@umn.edu 
#SBATCH --output=output/TestK5.out

cd $OLDHOME/welfare-meta-study
../julia-1.9.3/bin/julia -t 50 tests/test_standard_errors.jl
