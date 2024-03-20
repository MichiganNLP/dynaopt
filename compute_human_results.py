import csv
import ast
import numpy as np
files = ["aj.csv", "am.csv"]
num_models = 4
res_dics = []
for file in files:
    res_dic = {}
    with open(file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        header = reader.fieldnames
        for row in reader:
            prompt = row["Client Prompt"]
            responses = [ row[f"Response {i}"] for i in range(1, num_models) ] 
            answers = [ row[f"Answer {i}"] for i in range(1, num_models) ]
            keys = row["keys"]
            keys = ast.literal_eval(keys)
            for key, answer in zip(keys, answers):
                if key not in res_dic:
                    res_dic[key] = []
                res_dic[key].append(answer)
    res_dics.append(res_dic)
assert res_dics[0].keys() == res_dics[1].keys()
for key in res_dics[0].keys():
    print(key)
    print(res_dics[0][key])
    print(res_dics[1][key])
    list_0 = res_dics[0][key]
    list_1 = res_dics[1][key]
    list_0 = [ int(x) if x != "" else 0 for x in list_0 ]
    list_1 = [ int(x) if x != "" else 0 for x in list_1 ]
    print(np.average(list_0+list_1))
print("="*30)
for key in res_dics[0].keys():
    list_0 = res_dics[0][key]
    list_1 = res_dics[1][key]
    avgs = []
    for l0, l1 in zip(list_0, list_1):
        if l0 != "":
            l0 = int(l0)
        else:
            l0 = 0
        if l1 != "":
            l1 = int(l1)
        else:
            l1 = 0
        avg = (l0 + l1) / 2.0
        avgs.append(avg)
    print(key)
    print(np.average(avgs)/2.0)