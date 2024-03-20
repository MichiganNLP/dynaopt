import openai
import asyncio
from aiolimiter import AsyncLimiter
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
)
import random
from tqdm import tqdm
import fire
openai.api_key = ""
openai.api_type = ""
openai.api_base = ""
openai.api_version = ""
rate_limit = AsyncLimiter(50, 60)
@retry(wait=wait_fixed(2), stop=stop_after_attempt(6))
async def task(msg, MAX_GEN_LEN=40):
    async with rate_limit:
        prompt = msg["prompt"]
        try:
            result = openai.ChatCompletion.create(
                engine="gpt-35-turbo-16k", 
                max_tokens=MAX_GEN_LEN,
                messages=[{"role": "user", "content": prompt}]
            )
        except:
            return msg, None
        gen = result.choices[0].message['content'].strip()
        msg.update({"generated": gen})
        return msg, result
cost_dict = {
    'gpt-3.5-turbo-0301': (0.0015, 0.002),
    'gpt-3.5-turbo-0613': (0.0015, 0.002),
    'gpt-3.5-turbo-16k-0613': (0.0015, 0.002),
    'gpt-4-0314': (0.03, 0.06),
    'gpt-4-0613': (0.03, 0.06),
}
async def batch_llm_evaluate_reflection(exchanges, prompt_examples):
    messages = []
    for exchange in tqdm(exchanges):
        client = exchange["prompt"]
        counselor = exchange["generated"] 
        instruction = (
            "Evalulate the counselor's response to the client prompt. Specifically, evaluate the level of reflective listening employed in the counselor response. "
            "Reflective listening in motivational interviewing involves skillfully paraphrasing and summarizing a client's thoughts and feelings, fostering a deeper understanding and encouraging them to explore their own motivations for change. "
            "Your score should be 0 for non-reflections and 1 for simple reflections, and 2 for complex reflections. "
            "You will be give examples of non, simple, and complex reflections. "
        )
        complex_examples = [ x for x in prompt_examples if x["level"] == 2]
        simple_examples = [ x for x in prompt_examples if x["level"] == 1]
        non_examples = [ x for x in prompt_examples if x["level"] == 0 ]
        complex_idx = random.sample(range(len(complex_examples)), 2)
        complex_example1 = f"Client: {complex_examples[complex_idx[0]]['prompt']}\nCounselor: {complex_examples[complex_idx[0]]['response']}"
        complex_example2 = f"Client: {complex_examples[complex_idx[1]]['prompt']}\nCounselor: {complex_examples[complex_idx[1]]['response']}"
        simple_idx = random.sample(range(len(simple_examples)), 2)
        simple_example1 = f"Client: {simple_examples[simple_idx[0]]['prompt']}\nCounselor: {simple_examples[simple_idx[0]]['response']}"
        simple_example2 = f"Client: {simple_examples[simple_idx[1]]['prompt']}\nCounselor: {simple_examples[simple_idx[1]]['response']}"
        non_idx = random.sample(range(len(non_examples)), 2)
        non_example1 = f"Client: {non_examples[non_idx[0]]['prompt']}\nCounselor: {non_examples[non_idx[0]]['response']}"
        non_example2 = f"Client: {non_examples[non_idx[1]]['prompt']}\nCounselor: {non_examples[non_idx[1]]['response']}"
        evaluate_prompt = f"{instruction}\n\nComplex Reflection Example 1:\n{complex_example1}\nComplex Reflection Example 2:\n{complex_example2}\n\
Simple Reflection Example 1:\n{simple_example1}\nSimple Reflection Example 2:\n{simple_example2}\n\
Non-Reflection Example 1:\n{non_example1}\nNon-Reflection Example 2:\n{non_example2}\n\n\
Target Client: {client}\n\
Target Counselor: {counselor}\n\
Reflection Score: "
        dic = {
            "prompt": evaluate_prompt,
        }
        messages.append(dic)
    tasks = []
    for msg in tqdm(messages):
        tasks.append(task(msg, MAX_GEN_LEN=100))
    results = await asyncio.gather(*tasks)
    results = [x for x in results if x[1] is not None]
    collected_data = [x[0] for x in results]
    predictions = [x["generated"].strip() for x in collected_data]
    new = []
    for p in predictions:
        try:
            p = float(p)
        except:
            continue
        new.append(p)
    predictions = new
    return predictions
from experiment_util import get_data
from process_data import read_umich_pair
import json
from glob import glob
import numpy as np
def evaluate_reflection(dir="./voutputs"):
    train_data, dev_data, test_data = get_data("MI_rl", None, 200, False)
    non = read_umich_pair(True) + read_umich_pair(False)
    non_examples = [ x for x in non if x["level"] != 2]
    prompt_examples = []
    for t in train_data:
        if "level" in t and t["level"] == 2:
            prompt_examples.append(t)
    prompt_examples += non_examples
    train_client_utterances = [x["prompt"] for x in train_data]
    train_counselor_utterances = [x["response"] for x in train_data]
    files = glob(dir + "/**/generated.json")
    files = sorted(files)
    dic = {}
    for file in files:
        kind = file.split("_MI_rl_2023")[0].split("test_MI_rl_")[-1]
        if "supervised" in kind and "test" not in kind:
            continue
        if kind not in dic:
            dic[kind] =  []
        with open(file) as f:
            data = json.load(f)
            dic[kind].append(data)
    res_dic = {}
    for k, v in dic.items():
        print("*"*45)
        print(k)
        if k not in res_dic:
            res_dic[k] = []
        v = [v[0]] 
        for vv in v:            
            results = asyncio.run(batch_llm_evaluate_reflection(vv, prompt_examples))
            fold_resuts = []
            for vvv, r in zip(vv, results):
                rd = { "prompt": vvv["prompt"], "generated": vvv["generated"], "gt": vvv["response"], "score": r }
                fold_resuts.append(rd)
            res_dic[k].append(fold_resuts)
        avgs = [ np.mean([y["score"] for y in x]) for x in res_dic[k]]
        avg = np.mean(avgs)
        std = np.std(avgs)
        print("avg:", avg)
        res_file =  k + "_results.json"
        new = {"avgs": avgs, "avg": avg, "std": std, "results": res_dic[k]}
        with open(res_file, "w") as f:
            json.dump(new, f, indent=4)
async def batch_llm_evaluate_coherence(exchanges, prompt_examples):
    messages = []
    for exchange in tqdm(exchanges):
        client = exchange["prompt"]
        counselor = exchange["generated"] 
        instruction = (
            "Evalulate the counselor's response to the client prompt. Specifically, evaluate the coherence level in the counselor response. "
            "Rate the coherence of the counselor on a scale of 1 to 3 (1=not coherent at all, 2=somewhat coherent, 3=very coherent).  "
            "Coherent counselor responses should effectively address the client's concerns and maintain a logical flow of conversation. "
            "Output one of 1, 2, or 3."
        )
        evaluate_prompt = f"{instruction}\n\n\
Target Client: {client}\n\
Target Counselor: {counselor}\n\
Coherence Score: "
        dic = {
            "prompt": evaluate_prompt,
        }
        messages.append(dic)
    tasks = []
    for msg in tqdm(messages):
        tasks.append(task(msg, MAX_GEN_LEN=100))
    results = await asyncio.gather(*tasks)
    predictions = [x[0]["generated"].strip() if x[1] is not None else 0.0 for x in results]
    new = []
    for p in predictions:
        try:
            p = float(int(p))/3.0
        except:
            continue
        new.append(p)
    predictions = new
    return predictions
async def batch_llm_evaluate_fluency(exchanges, prompt_examples):
    messages = []
    for exchange in tqdm(exchanges):
        client = exchange["prompt"]
        counselor = exchange["generated"] 
        instruction = (
            "Evalulate the counselor's response to the client prompt. Specifically, evaluate the fluency level in the counselor response. "
            "Responses are rated on a scale from 1 to 3, where 1 indicates responses that lack fluency, "
            "2 signifies responses that are somewhat fluent, and 3 represents responses that are highly fluent and natural in their expression. "
            "Fluent counselor responses should convey information in a clear and easily understandable manner, ensuring effective communication with the client. "
            "Output one of 1, 2, or 3." 
        )
        evaluate_prompt = f"{instruction}\n\n\
Target Client: {client}\n\
Target Counselor: {counselor}\n\
Fluency Score: "
        dic = {
            "prompt": evaluate_prompt,
        }
        messages.append(dic)
    tasks = []
    for msg in tqdm(messages):
        tasks.append(task(msg, MAX_GEN_LEN=100))
    results = await asyncio.gather(*tasks)
    predictions = [x[0]["generated"].strip() if x[1] is not None else 0.0 for x in results]
    new = []
    for p in predictions:
        try:
            p = float(int(p))/3.0
        except:
            continue
        new.append(p)
    predictions = new
    return predictions
def evaluate(dir="./voutputs"):
    train_data, dev_data, test_data = get_data("MI_rl", None, 200, False)
    non = read_umich_pair(True) + read_umich_pair(False)
    non_examples = [ x for x in non if x["level"] != 2]
    prompt_examples = []
    for t in train_data:
        if "level" in t and t["level"] == 2:
            prompt_examples.append(t)
    prompt_examples += non_examples
    files = glob(dir + "/**/generated.json")
    files = sorted(files)
    dic = {}
    for file in files:
        kind = file.split("_MI_rl_2023")[0].split("test_MI_rl_")[-1]
        print(kind)
        if kind.split("/")[-1] not in [ "rl_scst_bandit", "con_scst_contextual", "rl_scst_bandit_weighted", "rl_scst_weighted"]:
            continue
        if kind not in dic:
            dic[kind] =  []
        with open(file) as f:
            data = json.load(f)
            dic[kind].append(data)
    print(dic.keys())
    res_dic = {}
    for k, v in dic.items():
        print("*"*45)
        print(k)
        if k not in res_dic:
            res_dic[k] = []
        v = [v[0]] 
        for vv in v:            
            vv = random.sample(vv, 100)
            coherence_results = asyncio.run(batch_llm_evaluate_coherence(vv, prompt_examples))
            fluency_results = asyncio.run(batch_llm_evaluate_fluency(vv, prompt_examples))
            fold_resuts = []
            for vvv, cr, fr in zip(vv, coherence_results, fluency_results):
                rd = { "prompt": vvv["prompt"], "generated": vvv["generated"], "gt": vvv["response"], "fluency": fr, "coherence": cr }
                fold_resuts.append(rd)
            res_dic[k].append(fold_resuts)
        f_avgs = [ np.mean([y["fluency"] for y in x]) for x in res_dic[k]]
        c_avgs = [ np.mean([y["coherence"] for y in x]) for x in res_dic[k]]
        f_avg = np.mean(f_avgs)
        c_avg = np.mean(c_avgs)
        f_std = np.std(f_avgs)
        c_std = np.std(c_avgs)
        print("f_avg:", f_avg)
        print("c_avg:", c_avg)
        res_file =  k + "_results_fc.json"
        new = {"f_avgs": f_avgs, "c_avgs": c_avgs, "f_avg": f_avg, "c_avg": c_avg, "f_std": f_std, "c_std": c_std, "results": res_dic[k]}
        with open(res_file, "w") as f:
            json.dump(new, f, indent=4)
if __name__ == "__main__":
    fire.Fire(evaluate)
