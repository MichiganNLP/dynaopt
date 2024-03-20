import csv 
import random
from tqdm import tqdm
import fire
import json
from glob import glob
import numpy as np
def make_sheet(dir="./voutputs"):
    test_kind = [ "con_scst_contextual_MI", "rl_scst_bandit_MI", "rl_scst_bandit_weighted_MI", "rl_scst_weighted_MI"]
    files = glob(dir + "/**/generated.json")
    files = sorted(files)
    print(files)
    dic = {}
    for file in files:
        test_file = file.replace("generated.json", "test_results.json")
        kind = "_".join(file.split("_")[:-7]).split("/")[-1]
        if kind not in test_kind:
            continue
        print(kind)
        with open(test_file) as f:
            t = json.load(f)
        if kind not in dic:
            dic[kind] =  []
        with open(file) as f:
            data = json.load(f)
            dic[kind].append(data)
    res_dic = {}
    for k, v in dic.items():
        v = v[0]
        for vv in v:
            prompt = vv["prompt"]
            response = vv["response"]
            generated = vv["generated"]
            if prompt not in res_dic:
                res_dic[prompt] = {}
            res_dic[prompt][k] = {}
            for key, val in vv.items():
                if key == "prompt":
                    continue
                res_dic[prompt][k][key] = val
    if True:
        filtered = {}
        chosen_keys = [ k for k,v in res_dic.items() if len(v.keys()) == 4]
        chosen_keys = random.sample(chosen_keys, 100)
        for k in chosen_keys:
            filtered[k] = res_dic[k]
    else:
        filtered = {}
        counter = 0
        idx_counter = 0
        idx_list = []
        for k, v in res_dic.items():
            print("="*30)
            print(k)
            print(f"{counter}/100 | {idx_counter}/{len(res_dic)}")
            print()
            for kk, vv in v.items():
                print(kk)
                print(vv["generated"])
                print()
            try:
                chose = int(input("1 = in / 0 = out: "))
            except:
                chose = 0
            if chose == -1:
                exit()
            if chose == 1:
                filtered[k] = v
                counter += 1
                idx_list.append(idx_counter)
            idx_counter += 1
            if counter == 100:
                break
        with open("./voutputs/idx_list.json", "w") as f:
            json.dump(idx_list, f)
    with open("./annotate_sheet.csv", "w") as fh:
        csvwriter = csv.writer(fh)
        csvwriter.writerow(["prompt", "Response 1", "Judgment 1", "Response 2", "Judgment 2", "Response 3", "Judgment 3", "Response 4", "Judgment 4", "gt", "keys"])
        for k,v in filtered.items():
            row = [k]
            keys = []
            for kk, vv in v.items():
                keys.append(kk)
            random.shuffle(keys)
            assert len(keys) == 4, f"keys should be 4 but {keys}"
            for key in keys:
                row.append(v[key]["generated"])
                row.append("")
            row.append(v[kk]["response"])
            row.append(keys)
            csvwriter.writerow(row)
if __name__ == "__main__":
    fire.Fire(make_sheet)
