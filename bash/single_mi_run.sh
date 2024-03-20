#!/bin/bash

# CUDA_VISIBLE_DEVICES=1,3 python supervised_train.py --experiment MI --num_epochs 5
#start_dir=./voutputs/supervised_MI_2023_09_14_17_34_32/supervised_MI_epochs2/
start_dir=./voutputs/supervised_MI_2023_09_20_11_28_27/supervised_MI_epochs5/
#CUDA_VISIBLE_DEVICES=2,3 python test_util.py --experiment MI_rl --model_start_dir $start_dir

# CUDA_VISIBLE_DEVICES=2,3 python rl_train.py --learning_mode bandit --seed 9  --experiment MI_rl --model_start_dir $start_dir

# CUDA_VISIBLE_DEVICES=2,3 python rl_train.py --learning_mode bandit_weighted --seed 10 --experiment MI_rl --model_start_dir $start_dir
# CUDA_VISIBLE_DEVICES=2,3 python con_rl_train.py --seed 10  --experiment MI_rl --model_start_dir $start_dir
# CUDA_VISIBLE_DEVICES=2,3 python rl_train.py --learning_mode weighted --seed 10 --experiment MI_rl --model_start_dir $start_dir
# CUDA_VISIBLE_DEVICES=2,3 python rl_train.py --learning_mode round --seed 10  --experiment MI_rl --model_start_dir $start_dir
# CUDA_VISIBLE_DEVICES=2,3 python rl_train.py --learning_mode bandit --seed 10  --experiment MI_rl --model_start_dir $start_dir


#CUDA_VISIBLE_DEVICES=1,3 python rl_train.py --learning_mode bandit_weighted --seed 2 --experiment MI_rl --model_start_dir $start_dir
# CUDA_VISIBLE_DEVICES=1,3 python rl_train.py --learning_mode weighted --seed 4 --experiment MI_rl --model_start_dir $start_dir

for i in {1..5}
do
   echo "Random Seed" $i
   CUDA_VISIBLE_DEVICES=2,3 python rl_train.py --learning_mode single --single_reward_idx 0 --seed $i --experiment MI_rl --model_start_dir $start_dir
   CUDA_VISIBLE_DEVICES=2,3 python rl_train.py --learning_mode single --single_reward_idx 1 --seed $i --experiment MI_rl --model_start_dir $start_dir
   CUDA_VISIBLE_DEVICES=2,3 python rl_train.py --learning_mode single --single_reward_idx 2 --seed $i --experiment MI_rl --model_start_dir $start_dir
   #CUDA_VISIBLE_DEVICES=2,3 python con_rl_train.py --seed $i  --experiment MI_rl --model_start_dir $start_dir
   #CUDA_VISIBLE_DEVICES=2,3 python rl_train.py --learning_mode weighted --seed $i --experiment MI_rl --model_start_dir $start_dir
   #CUDA_VISIBLE_DEVICES=2,3 python rl_train.py --learning_mode round --seed $i  --experiment MI_rl --model_start_dir $start_dir
   #CUDA_VISIBLE_DEVICES=2,3 python rl_train.py --learning_mode bandit --seed $i  --experiment MI_rl --model_start_dir $start_dir
done

python compute_stats.py --dir ./voutputs
