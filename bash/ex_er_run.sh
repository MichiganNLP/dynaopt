#!/bin/bash

#CUDA_VISIBLE_DEVICES=0,1 python test_util.py

#CUDA_VISIBLE_DEVICES=0,1 python rl_train.py --learning_mode "single" --single_reward_idx 0
#CUDA_VISIBLE_DEVICES=0,1 python rl_train.py --learning_mode "single" --single_reward_idx 1
#CUDA_VISIBLE_DEVICES=0,1 python rl_train.py --learning_mode "weighted"
#CUDA_VISIBLE_DEVICES=0,1 python rl_train.py --learning_mode "bandit"
#CUDA_VISIBLE_DEVICES=0,1 python rl_train.py --learning_mode "bandit_weighted"


# This is for the reptition
for i in {1..5}
do
    echo "Iter" $i
    #CUDA_VISIBLE_DEVICES=0,1 python rl_train.py --learning_mode "single" --single_reward_idx 0 --seed $i | tee logs/ex_er/single0_$i.log
    #CUDA_VISIBLE_DEVICES=0,1 python rl_train.py --learning_mode "single" --single_reward_idx 1 --seed $i | tee logs/ex_er/single1_$i.log
    CUDA_VISIBLE_DEVICES=0,1 python rl_train.py --learning_mode "weighted" --seed $i | tee logs/ex_er/weighted_$i.log
    CUDA_VISIBLE_DEVICES=0,1 python rl_train.py --learning_mode "bandit" --seed $i | tee logs/ex_er/bandit_$i.log
    CUDA_VISIBLE_DEVICES=0,1 python rl_train.py --learning_mode "bandit_weighted" --seed $i | tee logs/ex_er/bw_$i.log
done
