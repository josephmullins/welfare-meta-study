#!/bin/bash -l        
#SBATCH --time=0:10:00
#SBATCH --ntasks=100
#SBATCH --mem=120g
#SBATCH --tmp=120g
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=mullinsj@umn.edu 
#SBATCH --output=TestThreads.out

cd ~/welfare-meta-study
~/julia-1.9.3/bin/julia -t 20 scripts/test_threaded.jl
~/julia-1.9.3/bin/julia -t 40 scripts/test_threaded.jl
~/julia-1.9.3/bin/julia -t 90 scripts/test_threaded.jl
