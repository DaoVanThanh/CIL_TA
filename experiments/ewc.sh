#!/bin/bash

set -e

gpu=$1
seed=$2
tag=$3
dataset=$4
num_tasks=$5
nc_first_task=$6
network=$7
num_epochs=$8
lamb=$9
alpha=${10}

if [ "${dataset}" = "imagenet_subset_kaggle" ]; then
  clip=1.0
else
  clip=100.0
fi

exp_name="${tag}:lamb${lamb}:base"
result_path="results/${tag}/ewc_base_${lamb}_${alpha}_${seed}"
python src/main_incremental.py \
  --exp-name ${exp_name} \
  --gpu ${gpu} \
  --datasets ${dataset} \
  --num-tasks ${num_tasks} \
  --nc-first-task ${nc_first_task} \
  --network ${network} \
  --use-test-as-val \
  --lr 0.1 \
  --clipping ${clip} \
  --nepochs ${num_epochs} \
  --batch-size 128 \
  --seed ${seed} \
  --cache-first-task-model \
  --log disk \
  --results-path ${result_path} \
  --tags ${tag} \
  --approach ewc \
  --lamb ${lamb} \
  --alpha ${alpha}