#!/bin/bash
#SBATCH -N 1 # Number of nodes
#SBATCH -n 10 # Number of tasks (procs)
#SBATCH -p amdq_milan # Partition
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1000  
#SBATCH -A dm2s # account
#SBATCH -t 6:00:00 # walltime
#SBATCH -J QuadPost # jobname
#SBATCH -o quadpost.%j.o # output
#SBATCH -e quadpost.%j.e # error

for i in $(seq 1 10); do
  srun --exclusive -n1 -c1 python post_eval_probit.py $i &
done

wait