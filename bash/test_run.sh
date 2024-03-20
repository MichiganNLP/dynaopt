#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 python test_util.py --model_start_dir foutputs/scst_weighted_empathy_full_2023_05_06_12_52_07/pytorch_model.bin --experiment empathy_full
CUDA_VISIBLE_DEVICES=0,1 python test_util.py --model_start_dir foutputs/scst_weighted_empathy_full_2023_05_06_18_25_49/pytorch_model.bin --experiment empathy_full
CUDA_VISIBLE_DEVICES=0,1 python test_util.py --model_start_dir foutputs/scst_weighted_empathy_full_2023_05_07_00_00_18/pytorch_model.bin --experiment empathy_full
CUDA_VISIBLE_DEVICES=0,1 python test_util.py --model_start_dir foutputs/scst_weighted_empathy_full_2023_05_07_05_27_15/pytorch_model.bin --experiment empathy_full
CUDA_VISIBLE_DEVICES=0,1 python test_util.py --model_start_dir foutputs/scst_weighted_empathy_full_2023_05_07_16_59_47/pytorch_model.bin --experiment empathy_full




