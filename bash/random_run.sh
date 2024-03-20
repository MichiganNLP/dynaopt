#!/bin/bash


#CUDA_VISIBLE_DEVICES=0,1 python rl_train.py --learning_mode "single" --single_reward_idx 0
#CUDA_VISIBLE_DEVICES=0,1 python rl_train.py --learning_mode "single" --single_reward_idx 1
#CUDA_VISIBLE_DEVICES=0,1 python rl_train.py --learning_mode "weighted"
#CUDA_VISIBLE_DEVICES=0,1 python rl_train.py --learning_mode "bandit"
#CUDA_VISIBLE_DEVICES=0,1 python rl_train.py --learning_mode "bandit_weighted"

#CUDA_VISIBLE_DEVICES=2,3 python test_util.py --experiment empathy_full | tee zoutputs/logs/full/mle_test.log
#CUDA_VISIBLE_DEVICES=0,1 python rl_train.py --learning_mode "single" --experiment empathy_full --single_reward_idx 0 --seed 0 | tee foutputs/logs/single0_0.log
#CUDA_VISIBLE_DEVICES=0,1 python rl_train.py --learning_mode "single" --experiment empathy_full --single_reward_idx 1 --seed 0 | tee foutputs/logs/single1_0.log
#CUDA_VISIBLE_DEVICES=0,1 python rl_train.py --learning_mode "single" --experiment empathy_full --single_reward_idx 2 --seed 0 | tee foutputs/logs/single2_0.log
    
#CUDA_VISIBLE_DEVICES=0,1 python rl_train.py --learning_mode "weighted" --experiment empathy_full --seed 5 | tee foutputs/logs/weighted_5.log

# This is for the reptition
for i in {1..5}
do
   echo "Iter" $i
   #CUDA_VISIBLE_DEVICES=0,1 python rl_train.py --learning_mode "weighted" --experiment empathy_full --seed $i | tee foutputs/logs/weighted_$i.log
   #CUDA_VISIBLE_DEVICES=0,1 python rl_train.py --learning_mode "bandit" --experiment empathy_full --seed $i | tee foutputs/logs/bandit_$i.log
   CUDA_VISIBLE_DEVICES=0,1 python random_rl_train.py  --seed $i 
done

# CUDA_VISIBLE_DEVICES=2,3 python test_util.py --experiment empathy_full --model_start_dir coutputs/con_scst_contextual_empathy_full_2023_05_10_17_51_13/pytorch_model.bin
# CUDA_VISIBLE_DEVICES=2,3 python test_util.py --experiment empathy_full --model_start_dir coutputs/con_scst_contextual_empathy_full_2023_05_10_20_21_01/pytorch_model.bin
# CUDA_VISIBLE_DEVICES=2,3 python test_util.py --experiment empathy_full --model_start_dir coutputs/con_scst_contextual_empathy_full_2023_05_10_22_47_38/pytorch_model.bin
# CUDA_VISIBLE_DEVICES=2,3 python test_util.py --experiment empathy_full --model_start_dir coutputs/con_scst_contextual_empathy_full_2023_05_11_01_09_01/pytorch_model.bin
# CUDA_VISIBLE_DEVICES=2,3 python test_util.py --experiment empathy_full --model_start_dir coutputs/con_scst_contextual_empathy_full_2023_05_11_03_39_59/pytorch_model.bin
