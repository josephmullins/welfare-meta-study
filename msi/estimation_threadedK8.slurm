#!/bin/bash -l        
#SBATCH --time=36:00:00
#SBATCH --ntasks=80
#SBATCH --mem=250g
#SBATCH --tmp=250g
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=mullinsj@umn.edu 
#SBATCH --output=output/EstimationK8.out

cd ~/welfare-meta-study
~/julia-1.9.3/bin/julia -t 80 scripts/estimate_model_childsample_K8.jl
