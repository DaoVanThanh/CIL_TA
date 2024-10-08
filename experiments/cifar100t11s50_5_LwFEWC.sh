#!/bin/bash

#SBATCH --time=72:00:00   # walltime
#SBATCH --ntasks=3   # number of processor cores (i.e. tasks)
#SBATCH --gpus=1

set -e

# eval "$(conda shell.bash hook)"
# conda activate ta

num_tasks=11
nc_first_task=50
num_epochs=200
dataset=cifar100_icarl
network=resnet32
tag=cifar100t${num_tasks}s${nc_first_task}

lamb_lwf=10
lamb_ewc=5000

for seed in 0 1 2; do
  ./experiments/lwf_ewc.sh 0 ${seed} ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} ${num_epochs} ${lamb_lwf} ${lamb_ewc}
done
