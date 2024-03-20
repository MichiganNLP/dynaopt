#!/bin/bash

# CUDA_VISIBLE_DEVICES=1,2 python rl_train.py --learning_mode bandit --seed 4


# CUDA_VISIBLE_DEVICES=1,2 python rl_train.py --learning_mode bandit_weighted --seed 5
# CUDA_VISIBLE_DEVICES=1,2 python con_rl_train.py --seed 5
# CUDA_VISIBLE_DEVICES=1,2 python rl_train.py --learning_mode weighted --seed 5
# CUDA_VISIBLE_DEVICES=1,2 python rl_train.py --learning_mode round --seed 5 
# CUDA_VISIBLE_DEVICES=1,2 python rl_train.py --learning_mode bandit --seed 5
start_dir=./webnlg_outputs/supervised_webnlg_2023_09_21_16_28_10/supervised_webnlg_epochs2/

# CUDA_VISIBLE_DEVICES=1,3 python supervised_train.py --experiment webnlg


#CUDA_VISIBLE_DEVICES=1,3 python test_util.py --experiment webnlg --model_start_dir $start_dir

# for i in {6..10}
for i in {1..5}

do
   echo "Random Seed" $i
   CUDA_VISIBLE_DEVICES=1,3 python rl_train.py --learning_mode bandit_weighted --seed $i --experiment webnlg --model_start_dir $start_dir
   CUDA_VISIBLE_DEVICES=1,3 python con_rl_train.py --seed $i  --experiment webnlg --model_start_dir $start_dir
   CUDA_VISIBLE_DEVICES=1,3 python rl_train.py --learning_mode weighted --seed $i  --experiment webnlg --model_start_dir $start_dir
   CUDA_VISIBLE_DEVICES=1,3 python rl_train.py --learning_mode round --seed $i  --experiment webnlg --model_start_dir $start_dir
   CUDA_VISIBLE_DEVICES=1,3 python rl_train.py --learning_mode bandit --seed $i --experiment webnlg --model_start_dir $start_dir
done

python compute_stats.py --dir ./webnlg_outputs
