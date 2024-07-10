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
wu_epochs=${10:-0}

if [ "${dataset}" = "imagenet_subset_kaggle" ]; then
  clip=1.0
else
  clip=100.0
fi

if [ ${wu_epochs} -gt 0 ]; then
  exp_name="${tag}:lamb_${lamb}:mc:base:wu"
  result_path="results/${tag}/lwf_mc_base_wu_${lamb}_${seed}"
  python3 src/main_incremental.py \
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
    --log disk \
    --cache-first-task-model \
    --results-path ${result_path} \
    --tags ${tag} \
    --approach lwf \
    --mc \
    --lamb ${lamb} \
    --wu-nepochs ${wu_epochs} \
    --wu-lr 0.1 \
    --wu-fix-bn \
    --wu-scheduler onecycle \
    --wu-patience 50
else
  exp_name="${tag}:lamb_${lamb}:mc:base"
  result_path="results/${tag}/lwf_mc_base_${lamb}_${seed}"
  python3 src/main_incremental.py \
    --exp-name ${exp_name} \
    --gpu ${gpu} \
    --datasets ${dataset} \
    --num-tasks ${num_tasks} \
    --nc-first-task ${nc_first_task} \
    --network ${network} \
    --use-test-as-val \
    --clipping ${clip} \
    --nepochs ${num_epochs} \
    --batch-size 128 \
    --seed ${seed} \
    --log disk \
    --cache-first-task-model \
    --results-path ${result_path} \
    --tags ${tag} \
    --approach lwf \
    --mc \
    --lamb ${lamb}
fi
