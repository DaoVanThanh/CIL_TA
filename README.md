 # Repository
This repository is based on [FACIL](https://github.com/mmasana/FACIL).

# Objectives
This project aims to develop and implement novel Class Incremental Learning approach that effectively mitigate catastrophic forgetting while improving model adaptability to new classes.

# Installation
Install required dependencies:

    pip install -r requirements.txt

# Usage
    python src/main_incremental.py 
    --exp-name cifar100t10s10 
    --gpu 0 
    --datasets cifar100_icarl 
    --num-tasks 10
    --nc-first-task 10
    --gridsearch-tasks 10 
    --network resnet32 
    --lr 0.1 
    --clipping 100.0 
    --nepochs 200 
    --batch-size 128 
    --seed 0 
    --log disk tensorboard wandb
    --results-path results/cifar100t10s10/lwf_ewc 
    --tags cifar100t10s10 
    --approach lwf_ewc 
    --T 2 
    --lamb-lwf 10 
    --lamb-ewc 10000 
    --eval-on-train 
    --save-models 
    --num-exemplars 2000
    --exemplar-selection herding