import utils_misc, time, argparse, numpy as np, wandb
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig, BartModel
from transformers import AutoModel, AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration, T5Config, AutoModelForSeq2SeqLM
import utils_optim, utils_scoring, utils_rl, utils_timing
from datasets import load_dataset
from datetime import datetime
import torch
from tqdm import tqdm
import torch.nn.functional as F
import os
import json
from typing import Optional, List, Dict, Tuple
import fire
from process_data import *
from bandit_alg import Exp3
from model_reflection import ReflectionScoreDeployedCL
from model_empathy import Empathy
from peft_util import *
from random_util import set_seed
from test_util import *
from experiment_util import *
from harmonic_score import harmonic_score
from nltk import sent_tokenize
def scale_reward(history: List[float], reward: float):
    if history == []:
        return reward
    q20 = np.quantile(history, 0.2)
    q80 = np.quantile(history, 0.8)
    if reward < q20:
        reward = 0.0
    elif reward > q80:
        reward = 1.0
    else:
        reward = (reward - q20) / (q80 - q20)
    return reward
def to_vw_example_format(context, actions, cb_label=None):
    if cb_label is not None:
        chosen_action, cost, prob = cb_label
    example_string = ""
    example_string += "shared | f0={} f1={} f2={} r0={} r1={} r2={} \n".format(
        context["feature0"], context["feature1"], context["feature2"],
        context["scaled_reward0"], context["scaled_reward1"], context["scaled_reward2"]
    )
    for action in actions:
        if cb_label is not None and action == chosen_action:
            example_string += "0:{}:{} ".format(cost, prob)
        example_string += "|Action arm={} \n".format(action)
    return example_string[:-1]
def sample_custom_pmf(pmf):
    total = sum(pmf)
    scale = 1 / total
    pmf = [x * scale for x in pmf]
    draw = random.random()
    sum_prob = 0.0
    for index, prob in enumerate(pmf):
        sum_prob += prob
        if sum_prob > draw:
            return index, prob
import vowpalwabbit
def get_action(vw, context, actions):
    vw_text_example = to_vw_example_format(context, actions)
    print("CB Predict Example:", vw_text_example)
    pmf = vw.predict(vw_text_example)
    if False:
        print("pmf:", pmf)
    chosen_action_index, prob = sample_custom_pmf(pmf)
    return actions[chosen_action_index], prob, pmf
def run_rl_train(
    model_name: str = "t5-base",
    model_start_dir: Optional[str] = "moutputs/supervised_MI_2023_09_07_11_07_47/supervised_MI_epochs2/",
    max_seq_length: int = 90,
    max_output_length: int = 45, 
    scoring: str = "logsum",
    learning_rate: float = 1e-4,
    optimizer_name: str = "adam",
    train_batch_size: int = 1,
    num_runs: int = 10, 
    rl_run_size = 20,
    rl_validation_step = 10, 
    rl_validation_size = 200, 
    ckpt_every: int = 600,
    ckpt_lookback: int = 100,
    print_every: int = 200, 
    timings: bool = False,
    num_steps: int = 1000, 
    use_apex: bool = True,
    experiment = "MI_rl",
    learning_mode = "contextual",
    grad_accum_steps = 1,
    data_list: Optional[List] = None,
    lora=False,
    load_in_8bit=False,
    test=True,
    debug=False,
    single_reward_idx = 0,
    seed = 420691488,
    do_wandb = True,
    goals = None, 
    kl_coeff = 0.05,
    ):
    print("="*30)
    print("model_name: ", model_name)
    print("model_start_dir: ", model_start_dir)
    print("max_seq_length: ", max_seq_length)
    print("max_output_length: ", max_output_length)
    print("scoring: ", scoring)
    print("learning_rate: ", learning_rate)
    print("optimizer_name: ", optimizer_name)
    print("train_batch_size: ", train_batch_size)
    print("num_runs: ", num_runs)
    print("rl_run_size: ", rl_run_size)
    print("rl_validation_step: ", rl_validation_step)
    print("rl_validation_size: ", rl_validation_size)
    print("ckpt_every: ", ckpt_every)
    print("ckpt_lookback: ", ckpt_lookback)
    print("print_every: ", print_every)
    print("timings: ", timings)
    print("num_steps: ", num_steps)
    print("use_apex: ", use_apex)
    print("experiment: ", experiment)
    print("learning_mode: ", learning_mode)
    print("grad_accum_steps: ", grad_accum_steps)
    print("data_list: ", data_list)
    print("lora: ", lora)
    print("load_in_8bit: ", load_in_8bit)
    print("test: ", test)
    print("debug: ", debug)
    print("single_reward_idx: ", single_reward_idx)
    print("seed: ", seed)
    print("kl_coeff: ", kl_coeff)
    print("="*30)
    data_split = [0.8, 0.1, 0.1]
    train_data, dev_data, test_data = get_data(experiment, data_split, rl_validation_size, debug)
    set_seed(seed)
    if learning_mode not in ["bandit", "weighted", "bandit_weighted", "single", "contextual"]:
        raise ValueError("Unrecognized learning mode: %s" % learning_mode)
    gen_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    experiment = "con_scst_"+ learning_mode +"_"+ experiment
    if do_wandb:
        wandb.init(project="pair_scst")
        wandb.run.name = experiment
        wandb.run.save()
        wandb.config.update({"model_name": model_name,
                "model_start_dir": model_start_dir,
                "max_seq_length": max_seq_length,
                "max_output_length": max_output_length,
                "scoring": scoring,
                "learning_rate": learning_rate,
                "optimizer_name": optimizer_name,
                "train_batch_size": train_batch_size,
                "num_runs": num_runs,
                "rl_run_size": rl_run_size,
                "rl_validation_step": rl_validation_step,
                "rl_validation_size": rl_validation_size,
                "ckpt_every": ckpt_every,
                "ckpt_lookback": ckpt_lookback,
                "print_every": print_every,
                "timings": timings,
                "num_steps": num_steps,
                    "use_apex": use_apex,
                    "experiment": experiment,
                    "learning_mode": learning_mode,
                    "grad_accum_steps": grad_accum_steps,
                    "data_list": data_list,
                    "lora": lora,
                    "load_in_8bit": load_in_8bit,
                    "test": test,
                    "debug": debug,
                    "single_reward_idx": single_reward_idx,
                    "seed": seed,
                    "kl_coeff": kl_coeff,}
        )
    begin_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output_dir =  "./voutputs/%s_%s/" % (experiment, begin_time)
    os.makedirs(output_dir, exist_ok=True)
    utils_misc.DoublePrint(output_dir+"/log", show_timings=True)
    N_samples = num_runs
    if experiment.endswith("cnn_daily"):
        max_seq_length = 512
    if experiment.endswith("common_gen"):
        model_start_dir = "models/supervised_common_gen_epochs1"
    elif experiment.endswith("MI_rl"):
        pass
    print("model_start_dir: ", model_start_dir)
    model, tokenizer = get_model(model_name, model_start_dir, max_seq_length=max_seq_length, lora=lora, max_output_length=max_output_length)
    model.to(gen_device)
    ref_model, tokenizer = get_model(model_name, model_start_dir, max_seq_length=max_seq_length, lora=lora, max_output_length=max_output_length)
    ref_model.to(gen_device)
    ref_model.eval()
    ref_model.requires_grad_(False)
    use_apex = use_apex
    if use_apex:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    optimizer = utils_optim.build_optimizer(model, optimizer_name=optimizer_name, learning_rate=learning_rate)
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
    dataloader = DataLoader(dataset=train_data, batch_size=train_batch_size,\
        sampler=RandomSampler(train_data), drop_last=True, collate_fn=batch_collate)
    rl_val_dataloader = DataLoader(dataset=dev_data, batch_size=rl_run_size,\
        sampler=SequentialSampler(dev_data), drop_last=True, collate_fn=batch_collate)
    ckpter = utils_rl.RLModelCheckpoint(model, ckpt_every, ckpt_lookback, output_dir)
    printer = utils_rl.RLExamplePrinter(print_every, N_samples, print_source=False, print_edit=True)
    timer = utils_timing.TickTimer()
    thermostat = utils_rl.RLThermostat()
    rl_crit = utils_rl.ReinforceCriterion(model, tokenizer, optimizer, scaler, \
                                          use_apex=use_apex, ref_model=ref_model, kl_coeff=kl_coeff)
    scorers = get_scorers(experiment, learning_mode, single_reward_idx)
    if goals == None:
        goals = [ 0.9 for _ in scorers]
    assert(len(goals) == len(scorers))
    scorer = utils_scoring.ScorerWrapper(scorers, learning_mode = learning_mode, \
         scoring_method=scoring, max_batch_size=12, rl_validation_step=rl_validation_step)
    T_start, T_last_best = time.time(), time.time()
    temperature = 1.0
    scaler = torch.cuda.amp.GradScaler()
    STEPS = num_steps
    step_count = 0
    cont = True
    if "contextual" in learning_mode:
        bandit_history = []
        bandit_weight_history = []
        pmf_history = []
        vw = vowpalwabbit.Workspace(f"--cb_explore_adf   --cover 3", quiet=True)
        actions = [str(i) for i in range(len(scorer.scorers)+1)]
        chosen = None
        last_chosen = None
        rl_scorer_history = { k["name"]+"_scores":[] for k in scorer.scorers }
        bandit_pulls = { i:0 for i in range(len(scorer.scorers)+1) }
    else:
        bandit = None
        chosen = None
        last_chosen = None
    while cont:
        for ib, paragraphs in enumerate(tqdm(dataloader, position=0, leave=True, dynamic_ncols=True)):
            responses = paragraphs["responses"]
            prompts = paragraphs["prompts"]
            T_batch_start = time.time()
            gen_params = {"max_new_tokens": max_output_length, "early_stopping": True,  \
                    "do_sample": True, "num_return_sequences": num_runs, "temperature": temperature,
                    }
            with torch.cuda.amp.autocast():
                gen_input = tokenizer.batch_encode_plus(prompts, max_length=max_seq_length, \
                    return_tensors="pt", padding="longest", truncation=True)
                gen_input = {k: v.to(gen_device) for k, v in gen_input.items()}
                try:
                    gens_out = model.generate(input_ids=gen_input["input_ids"],\
                        decoder_start_token_id=tokenizer.bos_token_id,\
                        attention_mask=gen_input["attention_mask"], **gen_params)
                except:
                    print("Error in generation")
                    continue
                generateds = tokenizer.batch_decode(gens_out, skip_special_tokens=True)
                generateds = [ [ x.strip() for x in g.split("[CLS]")[:-1]] for g in generateds]
                cls_generateds = [ [ x.strip() + " [CLS]" for x in g] for g in generateds ]
                cls_generateds = [ " ".join(g) for g in cls_generateds]
                generateds = [ " ".join(g) for g in generateds]
                generateds = [ g.replace("<pad>", "").strip() for g in generateds]
                generateds = [g.replace("[CLS]", "").strip() for g in generateds]
                gens_out = tokenizer.batch_encode_plus(cls_generateds, max_length=max_output_length, \
                    return_tensors="pt", padding="longest", truncation=True)["input_ids"]
                timer.tick("sampled_generation")    
                prompts = [p for p in prompts for _ in range(num_runs)]
                responses = [r for r in responses for _ in range(num_runs)]
                scorer_returns = scorer.score(prompts, generateds, responses=responses, step_count=step_count, bandit=None, chosen=chosen)
                chosen = None 
                total_scores = torch.FloatTensor(scorer_returns["total_scores"]).cuda()
                batch_scores = total_scores.reshape(train_batch_size, N_samples)
                mean_scores = batch_scores.mean(dim=1)
                max_scores = torch.max(batch_scores, dim=1).values 
                unlooped_mean_scores = torch.repeat_interleave(mean_scores, N_samples)
                timer.tick("all_scores")
                normalized_rewards = (unlooped_mean_scores - total_scores)
                n_diff_pos, n_diff_neg = (normalized_rewards<-0.02).long().sum().item(), (normalized_rewards>0.02).long().sum().item()
                print("[%d samples] %d above avg and %d below avg" % (train_batch_size*N_samples, n_diff_pos, n_diff_neg))
                diversity = len(set(generateds)) / len(generateds)
                temperature = thermostat.log_diversity(diversity)
                loss = rl_crit(prompts, gens_out, normalized_rewards)
                loss = loss / grad_accum_steps
            if use_apex:
                scaler.scale(loss).backward()
                if (step_count + 1) % grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if (step_count + 1) % grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
                    optimizer.step()
                    optimizer.zero_grad()
            timer.tick("optim")
            batch_time = time.time() - T_batch_start
            log_obj = {"loss": loss, "max_scores": max_scores.mean().item(),\
                "temperature": temperature, "elem_per_sec": (len(generateds) / (batch_time+0.001)),\
                "diversity": diversity }
            log_obj.update({k: np.mean(v) for k, v in scorer_returns.items() if "_scores" in k or k in ["fluency_disc_val_f1"] or k in "response_fluency_disc_val_f1" })
            log_obj.update({"gen":generateds})
            if timings:
                timer.report()
            check_scores = { k["name"]+"_scores":[] for k in scorer.scorers }
            for k,v in scorer_returns.items():
                if k in check_scores:
                    check_scores[k].extend(v)
            check_means = []
            for k,v in check_scores.items():
                check_means.append(np.mean(v))
            current_score = np.mean(scorer_returns['raw_total_scores'])
            is_best = True 
            if is_best: 
                T_last_best = time.time()
            printer.tick(prompts, generateds, scorer_returns, responses, steps=step_count)
            step_count += 1
            if STEPS != None and step_count >= STEPS:
                cont = False
                break
            current_scores = { k["name"]+"_scores":[] for k in scorer.scorers }
            if step_count !=0 and  step_count % rl_validation_step == 0 and ("contextual" in learning_mode ):
                for ix, batch,  in enumerate(tqdm(rl_val_dataloader, position=0, leave=True, dynamic_ncols=True)):
                    responses = batch["responses"]
                    prompts = batch["prompts"]
                    T_batch_start = time.time()
                    gen_params = {"max_new_tokens": max_output_length , "early_stopping": True,  \
                            "do_sample": True, "num_return_sequences": 1, "temperature": temperature,
                            }
                    with torch.cuda.amp.autocast():
                        gen_input = tokenizer.batch_encode_plus(prompts, max_length=max_seq_length, \
                            return_tensors="pt", padding="longest", truncation=True)
                        gen_input = {k: v.to(gen_device) for k, v in gen_input.items()}
                        try:
                            gens_out = model.generate(input_ids=gen_input["input_ids"],\
                                decoder_start_token_id=tokenizer.bos_token_id,\
                                attention_mask=gen_input["attention_mask"], **gen_params)
                        except:
                            print("Error in generation")
                            continue
                        generateds = tokenizer.batch_decode(gens_out, skip_special_tokens=True)
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
                    scorer_returns = scorer.rl_score(prompts, generateds, responses=responses, step_count=step_count, bandit=None)
                    for k,v in scorer_returns.items():
                        if k in current_scores:
                            current_scores[k].extend(v)
                print("Mean Rewards", [ np.mean(v) for k,v in current_scores.items() ])
                scaled = []
                for k,v in rl_scorer_history.items():
                    HISTORY_SIZE = rl_validation_size
                    history = v[-HISTORY_SIZE:]
                    if history == []:
                        scaled.append(0.0)
                    else:
                        scaled.append(np.mean(current_scores[k])-np.mean(history))
                print("Mean Scaled Rewards", scaled)
                if last_chosen != None:
                    print("Last Chosen", last_chosen)
                    action = last_chosen[0]
                    prob = last_chosen[1]
                    score = scaled[0] 
                    print("Improvements Score", score)
                    log_obj.update({"improvement_score":score})
                    cost  = 1.0 - score 
                    parsed = to_vw_example_format(prev_context, actions, (action, cost, prob))
                    print("CB Training Example", parsed)
                    vw.learn(parsed)
                weights = scorer.weight_bandit.weights
                weights = weights / np.sum(weights) 
                context = {
                    f"feature{i}": weight for i, weight in enumerate(weights)
                }
                context2 ={
                    f"scaled_reward{i}": reward for i, reward in enumerate(scaled)
                }
                context.update(context2)
                prev_context = context
                action, prob, pmf = get_action(vw, context, actions)
                pmf_history.append(pmf)
                log_obj.update({"action":int(action), "prob":prob, "pmf":pmf})
                log_obj.update({"bandit_weights":weights.tolist()})
                chosen = (action, prob)
                print("chosen action, prob, pmf", action, prob, pmf)
                last_chosen = chosen
                bandit_pulls[int(last_chosen[0])] += 1
                bandit_history.append(int(last_chosen[0]))
                bandit_weight_history.append(weights.tolist())
                print(f"Step {step_count} / Chosen arm: {chosen[0]}")
                print("Bandit Pull:", bandit_pulls)
                for k,v in current_scores.items():
                    rl_scorer_history[k].extend(v)
            if do_wandb:
                wandb.log(log_obj)
        if STEPS == None:
            break
    model_file_name= model_name.replace("/", "_")
    save_name =  f"{experiment}_steps{STEPS}"
    model.save_pretrained(f"{output_dir}/{save_name}")
    tokenizer.save_pretrained(f"{output_dir}/{save_name}")
    if "contextual" in learning_mode:
        with open(f"{output_dir}/bandit_history.json", "w") as f:
            json.dump(bandit_history, f, indent=4)
        with open(f"{output_dir}/bandit_weight_history.json", "w") as f:
            json.dump(bandit_weight_history, f, indent=4)
        with open(f"{output_dir}/pmf_history.json", "w") as f:
            json.dump(pmf_history, f, indent=4)
    if test:
        run_test(model, tokenizer, test_data, test_batch_size=16, experiment=experiment, output_dir=output_dir)
    return model, tokenizer
if __name__ == "__main__":
    fire.Fire(run_rl_train)
