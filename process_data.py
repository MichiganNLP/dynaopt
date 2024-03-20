import csv
import json
import random
from itertools import combinations, product, chain  
import os
from definitions import ROOT_DIR
id2header = { 0: "prompt", 1:"hq1", 2:"hq2", 3:"mq1", 4:"lq1", 5:"lq2", 6:"lq3", 7:"lq4", 8:"lq5", 9:"topic"}
def flatten(l, ltypes=(list, tuple)):    
    ltype = type(l)    
    l = list(l)    
    i = 0    
    while i < len(l):    
        while isinstance(l[i], ltypes):    
            if not l[i]:    
                l.pop(i)    
                i -= 1     
                break    
            else:    
                l[i:i + 1] = l[i]    
        i += 1    
    return ltype(l)    
def generate_combs_matched(dic):
    prompt = [dic["prompt"]]    
    hq = combinations([dic["hq1"], dic["hq2"]],1)    
    mq = combinations([dic["mq1"]],1)    
    lq = combinations([dic["lq1"],dic["lq2"], dic["lq3"], dic["lq4"], dic["lq5"] ], 2)    
    products = [ list(flatten(prod)) for prod in product(prompt, hq, mq, lq) ]    
    return products
def generate_combs_pair(dic, balanced_sampling=False):
    prompt = [dic["prompt"]]    
    hq = combinations([dic["hq1"], dic["hq2"]],1)    
    mq = combinations([dic["mq1"]],1)    
    lq = combinations([dic["lq1"],dic["lq2"], dic["lq3"], dic["lq4"], dic["lq5"] ],1)    
    topic = dic["topic"]
    hq_products = [ list(flatten(prod)) for prod in product(prompt, hq) ]    
    mq_products = [ list(flatten(prod)) for prod in product(prompt, mq) ]
    lq_products = [ list(flatten(prod)) for prod in product(prompt, lq) ]
    hq_dics = [ {"prompt": prod[0], "response": prod[1], "level":2, "anti_response": random.sample(lq_products, 1)[0][1], "anti_level":0, "topic":topic } for prod in hq_products ]
    mq_dics = [ {"prompt": prod[0], "response": prod[1], "level":1, "anti_response": random.sample(lq_products, 1)[0][1], "anti_level":0, "topic":topic  } for prod in mq_products ]
    lq_dics = [ {"prompt": prod[0], "response": prod[1], "level":0, "anti_response": random.sample(hq_products, 1)[0][1], "anti_level":2, "topic":topic  } for prod in lq_products ]
    lq_dics[0]["t_behavior"] = "advice"
    lq_dics[1]["t_behavior"] = "question"
    lq_dics[2]["t_behavior"] = "NA"
    lq_dics[3]["t_behavior"] = "NA"
    lq_dics[4]["t_behavior"] = "NA"
    if balanced_sampling:
        hq_dics = random.sample(hq_dics, 1) 
        mq_dics = random.sample(mq_dics, 1) 
        lq_dics = random.sample(lq_dics, len(hq_dics))
    return hq_dics + mq_dics + lq_dics
def read_umich(path=os.path.join(ROOT_DIR, "data", "topic_clean_umich.csv")):
    """
    Generic function for reading raw UMICH data
    """
    with open(path, "r") as f:
        reader = csv.reader(f)
        data = list(reader)
    new_data = []
    for row in data:
        assert(len(row) == 10)
        dic = {}
        for i,r in enumerate(row):
            dic[id2header[i]] = r
        new_data.append(dic)
    return new_data
def read_umich_matched(use_drug_cessation=True):
    """
    Function for processing raw UMICH data into PAIR training format
    """
    data = read_umich()
    new_data = []
    for dat in data:
        if use_drug_cessation:
            if dat["topic"] == "drug cessation":
                pair_dat = generate_combs_matched(dat)
                new_data += pair_dat
        else:
            if dat["topic"] != "drug cessation":
                pair_dat = generate_combs_matched(dat)
                new_data += pair_dat
    return new_data
def read_umich_pair(use_drug_cessation=True, balanced_sampling=False, path=os.path.join(ROOT_DIR, "data", "topic_clean_umich.csv")):
    """
    Function for processing raw UMICH data into MMLE/RL training format
    """
    data = read_umich(path=path)
    new_data = []
    for dat in data:
        if use_drug_cessation:
            if dat["topic"] == "drug cessation":
                pair_dat = generate_combs_pair(dat,balanced_sampling)
                new_data += pair_dat
        else:
            if dat["topic"] != "drug cessation":
                pair_dat = generate_combs_pair(dat, balanced_sampling)
                new_data += pair_dat
    return new_data
def read_anno():
    """
    Generic function for reading raw ANNO data
    """
    with open(os.path.join(ROOT_DIR, "data", "gen_annoMI.json"), "r") as f:
        data = json.load(f)
    return data
def read_anno_pair(use_drug_cessation=True):
    """
    Function for processing raw ANNO data into MMLE/RL training format
    """
    data = read_anno()
    new_data = []
    for dat in data:
        if use_drug_cessation:
            if dat["response"]["collapsed_topic"] == "drug cessation":
                dic = {"prompt": " ".join([ x["fastpunct_utterance_text"] for x in dat["prompt"]]), "response": dat["response"]["fastpunct_utterance_text"], \
                "level": 2 if dat["response"]["main_therapist_behaviour"] == "reflection" else 0}
                new_data.append(dic)
        else:
            if dat["response"]["collapsed_topic"] != "drug cessation":
                dic = {"prompt": " ".join([ x["fastpunct_utterance_text"] for x in dat["prompt"]]), "response": dat["response"]["fastpunct_utterance_text"], \
                "level": 2 if dat["response"]["main_therapist_behaviour"] == "reflection" else 0}
                new_data.append(dic)
    return new_data
import csv
def read_cc():
    """
    Generic function for reading raw CC data
    """
    with open(os.path.join(ROOT_DIR, "data", "cc_mi.csv"), "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        data = list(reader)
    new_data = []   
    for row in data:
        dic = {}
        for i,r in enumerate(row):
            dic[header[i]] = r
        new_data.append(dic)
    data = new_data
    return data
def read_cc_pair():
    """
    Function for processing raw CC data into MMLE/RL training format
    """
    data = read_cc()
    new_data = []
    for i, dat in enumerate(data):
        if dat["author"] == "speaker":
            if i != len(data) - 1:
                if data[i+1]["author"] == "listener":
                    label = data[i+1]["final agreed label"]
                    if label == "Complex Reflection":
                        level = 2
                    elif label == "Simple Reflection":
                        level = 1
                    else:
                        level = 0
                    dic = {"prompt": dat["text"], "response": data[i+1]["text"], "level": level } 
                    new_data.append(dic)
    return new_data
def read_openai():
    with open(os.path.join(ROOT_DIR, "data", "openai.jsonl"), "r") as f:
        data = [json.loads(line) for line in f]
    return data
def read_openai_pair():
    return read_openai()
import csv
def read_csv_file(filename):
    with open(filename, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        data = []
        for row in csv_reader:
            data.append(row)
        return data
def get_empathy_reddit():
    ER = "data/emotional-reactions-reddit.csv"
    EP = "data/explorations-reddit.csv"
    IP = "data/interpretations-reddit.csv"
    er_data = read_csv_file(ER)
    ep_data = read_csv_file(EP)
    ip_data = read_csv_file(IP)
    data = er_data + ep_data + ip_data
    data = [x for x in data if int(x["level"]) == 2]
    for item in data:
        item["prompt"] = item["seeker_post"]
        item["response"] = item["response_post"]
    new_data = []
    prompts = set()
    for item in data:
        if item["prompt"] not in prompts:
            prompts.add(item["prompt"])
            new_data.append(item)
    data = new_data
    return data
def get_peer_reddit():
    with open("./data/reddit_peer_filtered.json", "r") as f:
        data = json.load(f)
    return data
def get_cnn_daily():
    with open("./data/cnn_daily.json", "r") as f:
        data = json.load(f)
    return data
def get_common_gen():
    with open("./data/common_gen.json", "r") as f:
        data_dic = json.load(f)
    return data_dic
def main():
    from transformers import BartTokenizer
    barto = BartTokenizer.from_pretrained("facebook/bart-base")
    data = get_peer_reddit()
    lens = []
    for dat in data:
        lens.append(len(barto.tokenize(dat["response"])))
    import numpy as np
    print(np.mean(lens))
    print(np.std(lens))
    print(np.median(lens))
if __name__ == "__main__":
    main()  
