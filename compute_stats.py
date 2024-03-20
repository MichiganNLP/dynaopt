import json
import fire
from glob import glob
import numpy as np
def main(dir = "./voutputs"):
    files = glob(dir + "/**/test_results.json")
    files = sorted(files)
    dic = {}
    for file in files:
        kind = file.split("_MI_rl_2023")[0].split("test_MI_rl_")[-1]
        print(kind)
        if kind not in dic:
            dic[kind] =  {}
        with open(file) as f:
            data = json.load(f)
            for d in data:
                if len(list(d.values())[0]) != 2:
                    continue
                k = list(d.keys())[0]
                if k not in dic[kind]:
                    dic[kind][k] = []
                dic[kind][k].append(list(d.values())[0])
    for kind in dic:
        print("*"*45)
        print(kind)
        for k in dic[kind]:
            print("="*30)
            print(k)
            for i in range(len(dic[kind][k][0])):
                arr = ([x[i] for x in dic[kind][k]])
                print(len(arr))
                print(arr)
                print(np.mean(arr), np.std(arr))
        print()
if __name__ == "__main__":
    fire.Fire(main)