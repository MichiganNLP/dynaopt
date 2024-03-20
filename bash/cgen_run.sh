#!/bin/bash

#CUDA_VISIBLE_DEVICES=2,3 python test_util.py ---experiment common_gen

for i in {1..5}
do
    echo "Iter" $i
    #CUDA_VISIBLE_DEVICES=2,3 python rl_train.py --learning_mode "single" --single_reward_idx 0 --experiment cnn_daily 
    #CUDA_VISIBLE_DEVICES=2,3 python rl_train.py --learning_mode "single" --single_reward_idx 1 --experiment cnn_daily
    CUDA_VISIBLE_DEVICES=2,3 python rl_train.py --learning_mode "weighted" --experiment common_gen  --seed $i | tee logs/cgen/weighted_$i.log
    CUDA_VISIBLE_DEVICES=2,3 python rl_train.py --learning_mode "bandit"   --experiment common_gen  --seed $i | tee logs/cgen/bandit_$i.log
    CUDA_VISIBLE_DEVICES=2,3 python rl_train.py --learning_mode "bandit_weighted"  --experiment common_gen  --seed $i | tee logs/cgen/bw_$i.log
done

# for i in {1..5}
# do
#     CUDA_VISIBLE_DEVICES=0,1 python rl_train.py --learning_mode "single" --single_reward_idx 0 --seed $i --experiment cnn_daily
#     CUDA_VISIBLE_DEVICES=0,1 python rl_train.py --learning_mode "single" --single_reward_idx 1 --seed $i --experiment cnn_daily
#     CUDA_VISIBLE_DEVICES=0,1 python rl_train.py --learning_mode "weighted" --seed $i --experiment cnn_daily
#     CUDA_VISIBLE_DEVICES=0,1 python rl_train.py --learning_mode "bandit" --seed $i --experiment cnn_daily
#     CUDA_VISIBLE_DEVICES=0,1 python rl_train.py --learning_mode "bandit_weighted" --seed $i --experiment cnn_daily
# done
