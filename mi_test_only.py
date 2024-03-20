import glob, os
models = glob.glob("./voutputs_231002/**/*steps*")
print(models, len(models))
for model in models:
   command = f"CUDA_VISIBLE_DEVICES=2,3 python test_util.py --experiment MI_rl --model_start_dir {model}"
   print(command)
   os.system(command)
