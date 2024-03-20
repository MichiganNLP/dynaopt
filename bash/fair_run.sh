#!/bin/bash


CUDA_VISIBLE_DEVICES=0,1 python rl_train.py --learning_mode "weighted" --experiment empathy_full --num_steps 20000 --seed 0 
CUDA_VISIBLE_DEVICES=0,1 python rl_train.py --learning_mode "bandit_weighted" --experiment empathy_full --num_steps 20000 --seed 0 

