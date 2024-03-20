#!/bin/bash

#CUDA_VISIBLE_DEVICES=2,3 python test_util.py --model_start_dir outputs/supervised_cnn_daily_2023_05_02_15_01_11/supervised_cnn_daily_epochs1/ --experiment cnn_daily

for i in {1..5}
do
    echo "Iter" $i
    #CUDA_VISIBLE_DEVICES=2,3 python rl_train.py --learning_mode "single" --single_reward_idx 0 --experiment cnn_daily 
    #CUDA_VISIBLE_DEVICES=2,3 python rl_train.py --learning_mode "single" --single_reward_idx 1 --experiment cnn_daily
    CUDA_VISIBLE_DEVICES=2,3 python rl_train.py --learning_mode "weighted" --experiment cnn_daily  --seed $i | tee logs/summary/weighted_$i.log
    CUDA_VISIBLE_DEVICES=2,3 python rl_train.py --learning_mode "bandit"   --experiment cnn_daily  --seed $i | tee logs/summary/bandit_$i.log
    CUDA_VISIBLE_DEVICES=2,3 python rl_train.py --learning_mode "bandit_weighted"  --experiment cnn_daily  --seed $i | tee logs/summary/bw_$i.log
done

# for i in {1..5}
# do
#     CUDA_VISIBLE_DEVICES=0,1 python rl_train.py --learning_mode "single" --single_reward_idx 0 --seed $i --experiment cnn_daily
#     CUDA_VISIBLE_DEVICES=0,1 python rl_train.py --learning_mode "single" --single_reward_idx 1 --seed $i --experiment cnn_daily
#     CUDA_VISIBLE_DEVICES=0,1 python rl_train.py --learning_mode "weighted" --seed $i --experiment cnn_daily
#     CUDA_VISIBLE_DEVICES=0,1 python rl_train.py --learning_mode "bandit" --seed $i --experiment cnn_daily
#     CUDA_VISIBLE_DEVICES=0,1 python rl_train.py --learning_mode "bandit_weighted" --seed $i --experiment cnn_daily
# done
