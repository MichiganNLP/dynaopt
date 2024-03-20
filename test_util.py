import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import fire
from tqdm import tqdm
import json
from datetime import datetime
import numpy as np
import utils_optim, utils_scoring, utils_rl, utils_timing
from model_reflection import ReflectionScoreDeployedCL
from model_empathy import Empathy
from experiment_util import *
from random_util import set_seed
from nltk import sent_tokenize
def run_test(
    model=None,
    tokenizer=None,
    data=None,
    test_batch_size=8,
    experiment = "empathy_EX_ER",
    scoring="logsum",
    output_dir="outputs/test",
    seed = 420691488,
    ):
    set_seed(seed)
    gen_device = model.device
    def batch_collate(inps):
        batch_paras = []
        batch_labels = []
        batch_responses = [] 
        for inp in inps:
            text = inp["prompt"] + " [SEP] " 
            batch_paras.append(text)
            batch_responses.append(inp["response"])
        return {"prompts": batch_paras,
                "responses": batch_responses
                }
    dataloader = DataLoader(dataset=data, batch_size=test_batch_size,\
        sampler=SequentialSampler(data), drop_last=True, collate_fn=batch_collate)
    gen_params = {"max_new_tokens": 100, "early_stopping": True,  \
            "do_sample": True, "num_return_sequences": 1, "temperature": 1.0,
            }
    scorers = get_scorers(experiment, None, None, False)
    scorer = utils_scoring.ScorerWrapper(scorers, scoring_method=scoring, max_batch_size=12)
    results = []
    outputs = []
    for ib, paragraphs in enumerate(tqdm(dataloader, position=0, leave=True, dynamic_ncols=True)):
        responses = paragraphs["responses"]
        prompts = paragraphs["prompts"]
        gen_params = {"max_new_tokens": model.config.max_output_length, "early_stopping": True,  \
                "do_sample": True, "num_return_sequences": 1, "temperature": 0.5,
                }
        with torch.cuda.amp.autocast():
            gen_input = tokenizer.batch_encode_plus(prompts, max_length=model.config.max_length, \
                return_tensors="pt", padding="longest", truncation=True)
            gen_input = {k: v.to(gen_device) for k, v in gen_input.items()}
            try:
                gens_out = model.generate(input_ids=gen_input["input_ids"],\
                        decoder_start_token_id=tokenizer.bos_token_id,\
                        attention_mask=gen_input["attention_mask"], **gen_params)
            except:
                print("Error generating")
                continue
            generateds = tokenizer.batch_decode(gens_out, skip_special_tokens=True)
            """
            special segment begin
            """
            cut_generateds = [ [ x.strip() for x in g.split("[CLS]")[:-1]] for g in generateds]
            new = []
            for c, g in zip(cut_generateds, generateds):
                if c == []:
                    new.append([g])
                else:
                    new.append(c)
            generateds = [ " ".join(g) for g in new]
            generateds = [ g.replace("<pad>", "").strip() for g in generateds]
            generateds = [g.replace("[CLS]", "").strip() for g in generateds]
            """
            special segment end
            """
            scorer_returns = scorer.rl_score(prompts, generateds, responses=responses)
            results.append(scorer_returns)
            for p, g, r in zip(prompts, generateds, responses):
                outputs.append({"prompt": p, "generated": g, "response": r})
    res_dict = {}
    for k,v in results[0].items():
        res_dict[k] = []
    for r in results:
        for k,v in r.items():
            if k in res_dict:
                res_dict[k] += v
            else:
                res_dict[k] = v
    for k,v in res_dict.items():
        assert len(v) == len(outputs) 
        for i,o in enumerate(outputs):
            o[k] = v[i]
    with open(output_dir + "/generated.json", "w") as f:
        json.dump(outputs, f, indent=4)
    res = []
    for k,v in res_dict.items():
        agg = {k: [np.mean(v), np.std(v)]}
        res.append(agg)
        print(agg)
    res.append(res_dict)
    with open(output_dir + "/test_results.json", "w") as f:
        json.dump(res, f, indent=4)
    return
def read_jsonl(path, line_length=9):
    with open(path, "r") as f:
        lines = f.readlines()
    lines = [lines[i:i+line_length] for i in range(0, len(lines), line_length)]
    data = []
    for l in lines:
        line = "".join(l)
        data.append(json.loads(line))
    return data
def run_only_test(
    model_name: str = "t5-base",
    model_start_dir: Optional[str] = "moutputs/supervised_MI_2023_09_07_11_07_47/supervised_MI_epochs2/",
    test_batch_size: int = 16,
    experiment = "MI_rl",
    debug: bool = False,
    lora: bool = False,
    seed: int = 420691488,
    ):
    if experiment == "common_gen":
        model_start_dir = "models/supervised_common_gen_epochs1"
    model, tokenizer = get_model(model_name, model_start_dir, max_seq_length=90, lora=lora)
    gen_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(gen_device)
    data_split = [0.8, 0.1, 0.1]
    train_data, dev_data, test_data = get_data(experiment, data_split, -1, debug )
    begin_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    model_type = [ x for x in model_start_dir.split("/") if x != "" and x!="."][1]
    output_dir =  "./voutputs/test_%s_%s_%s/" % (experiment, model_type, begin_time)
    os.makedirs(output_dir, exist_ok=True)
    run_test(model=model, tokenizer=tokenizer, data=test_data,\
        test_batch_size=test_batch_size, experiment = experiment,\
        output_dir=output_dir, seed=seed)
def run_only_naturalness_test(
    model_name: str = "t5-base",
    model_start_dir: Optional[str] = "models/supervised/model",
    test_batch_size: int = 8,
    test_generation_path: str = None,
    experiment = "empathy_EX_ER",
    debug: bool = False,
    lora: bool = False,
    seed: int = 420691488,
    ):
    if test_generation_path is None:
        test_generation_path = model_start_dir + "/generated.jsonl"
        if not os.path.exists(test_generation_path):
            test_generation_path = model_start_dir + "/generated.json"
    if test_generation_path.endswith(".jsonl"):
        test_data = read_jsonl(test_generation_path)
    else:
        with open(test_generation_path, "r") as f:
            test_data = json.load(f)
    for t in test_data:
        t["prompt"] = t["prompt"].replace("[SEP]", "").strip()
    begin_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output_dir =  "/".join(test_generation_path.split("/")[:-1]) + "/"
    def batch_collate(inps):
        batch_paras = []
        batch_generateds = []
        batch_responses = [] 
        for inp in inps:
            text = inp["prompt"] + " [SEP] " 
            batch_paras.append(text)
            batch_responses.append(inp["response"])
            batch_generateds.append(inp["generated"])
        return {"prompts": batch_paras,
                "responses": batch_responses,
                "generateds": batch_generateds}
    dataloader = DataLoader(dataset=test_data, batch_size=test_batch_size,\
        sampler=SequentialSampler(test_data), drop_last=True, collate_fn=batch_collate)
    scorers = get_naturalness_scorers(None, None)
    scorer = utils_scoring.ScorerWrapper(scorers, scoring_method="logsum", max_batch_size=12)
    results = []
    outputs = []
    for ib, paragraphs in enumerate(tqdm(dataloader, position=0, leave=True, dynamic_ncols=True)):
        responses = paragraphs["responses"]
        prompts = paragraphs["prompts"]
        generateds = paragraphs["generateds"]
        scorer_returns = scorer.rl_score(prompts, generateds, responses=responses)
        results.append(scorer_returns)
        for p, g, r in zip(prompts, generateds, responses):
            outputs.append({"prompt": p, "generated": g, "response": r})
    res_dict = {}
    for k,v in results[0].items():
        res_dict[k] = []
    for r in results:
        for k,v in r.items():
            if k in res_dict:
                res_dict[k] += v
            else:
                res_dict[k] = v
    for k,v in res_dict.items():
        assert len(v) == len(outputs) 
        for i,o in enumerate(outputs):
            o[k] = v[i]
    res = []
    for k,v in res_dict.items():
        agg = {k: [np.mean(v), np.std(v)]}
        res.append(agg)
        print(agg)
    res.append(res_dict)
    return
from model_multi import distinct_n_sentence_level
def run_corpus_distinct(
    model_start_dir: Optional[str] = "models/supervised/model",
    test_generation_path: str = None
    ):
    if test_generation_path is None:
        test_generation_path = model_start_dir + "/generated.jsonl"
        if not os.path.exists(test_generation_path):
            test_generation_path = model_start_dir + "/generated.json"
    if test_generation_path.endswith(".jsonl"):
        test_data = read_jsonl(test_generation_path)
    else:
        with open(test_generation_path, "r") as f:
            test_data = json.load(f)
    for t in test_data:
        t["prompt"] = t["prompt"].replace("[SEP]", "").strip()
    generateds = [t["generated"] for t in test_data]
    joined_generateds = " ".join(generateds)
    dis1 = distinct_n_sentence_level(joined_generateds.split(), 1)
    dis2 = distinct_n_sentence_level(joined_generateds.split(), 2)
    print("distinct-1: ", dis1)
    print("distinct-2: ", dis2)
    return
if __name__ == "__main__":
    fire.Fire(run_only_test)
