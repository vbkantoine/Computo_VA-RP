#!/bin/bash
#SBATCH -N 1 # Number of nodes
#SBATCH -n 100 # Number of tasks (procs)
#SBATCH -p amdq_milan # Partition
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1000  
#SBATCH -A dm2s # account
#SBATCH -t 6:00:00 # walltime
#SBATCH -J Probit # jobname
#SBATCH -o probit.%j.o # output
#SBATCH -e probit.%j.e # error

for i in $(seq 1 100); do
  srun --exclusive -n1 -c1 python probit_exp.py $i &
done

wait