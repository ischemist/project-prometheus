#!/bin/sh
#SBATCH -J h2o_sp
#SBATCH --output sbatch.out
#SBATCH --error sbatch.err
#SBATCH --time 01:00:00
#SBATCH --qos=premium
#SBATCH --account=m410
#SBATCH --ntasks=16
#SBATCH --nodes=1

module load orca

$(which orca) h2o.inp > h2o.out
