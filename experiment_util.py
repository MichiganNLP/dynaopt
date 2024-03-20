from typing import Optional, List, Dict, Tuple
from model_reflection import ReflectionScoreDeployedCL
from model_empathy import Empathy
from model_summary import Summary
from process_data import *
from prepare_webnlg_data import *
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig, BartModel
from transformers import AutoModel, AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration, T5Config, AutoModelForSeq2SeqLM
from peft_util import *
from model_multi import Multi
from model_webnlg import WebNLG
import torch
def get_model(
    model_name: str = "t5-base",
    model_start_dir: Optional[str] = None,
    load_in_8bit: bool = False,
    lora: bool = False,
    max_seq_length: int = 90,
    max_output_length: int = 90,
    ):
    if model_start_dir is not None:
        if model_start_dir.endswith(".bin"):
            model = T5ForConditionalGeneration.from_pretrained("t5-base") 
            tokenizer = T5Tokenizer.from_pretrained("t5-base")
            model.load_state_dict(torch.load(model_start_dir))
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_start_dir) 
            tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        if "bart" in model_name:
            tokenizer = BartTokenizer.from_pretrained(model_name)
            model = BartForConditionalGeneration.from_pretrained(model_name)
        elif "t5" in model_name:
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            model = T5ForConditionalGeneration.from_pretrained(model_name)
        elif "llama"in model_name:
            tokenizer = AutoTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
            model = AutoModelForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")
        elif False:
            print("To be implemented")
    model.config.max_length = max_seq_length
    tokenizer.model_max_length = max_seq_length
    model.config.max_output_length = max_output_length
    if lora:                                                                                                                           
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        model = make_lora_model(model, lora_r = 8, lora_alpha = 16, \
                lora_dropout = 0.05, lora_target_modules = [ "q_proj", "v_proj"]) 
    return model, tokenizer
def get_data(experiment: str, 
    data_split: Optional[List] = None,
    rl_validation_size = -1,
    debug=False,
    supervised=False,
    ):
    if data_split is None:
        data_split = [0.8, 0.1, 0.1]
    if experiment == "empathy_EX_ER":
        data = get_peer_reddit()
    elif experiment == "MI":
        data = read_umich_pair(True) + read_umich_pair(False)
        data = [ x for x in data if x["level"] == 2]
    elif experiment == "MI_rl":
        data_split = [0.5, 0.1, 0.4]
        data = read_umich_pair(True) + read_umich_pair(False) 
        with open("data/wellvita_dataset.json") as f:
            data += json.load(f)
        new_data = []
        prompts = set()
        for d in data:
            if d["prompt"] not in prompts:
                new_data.append(d)
                prompts.add(d["prompt"])
        data = new_data
    elif experiment == "empathy_reddit":
        data = get_empathy_reddit()
    elif experiment == "cnn_daily":
        data = get_cnn_daily()
    elif experiment == "common_gen":
        data = get_common_gen()
    elif experiment == "empathy_full":
        data = get_peer_reddit() 
    elif experiment == "webnlg":
        data = get_webnlg_data(data_name="webnlg", supervised=supervised)
    if experiment != "cnn_daily" and experiment != "common_gen" and experiment != "webnlg":
        train_data = data[:int(len(data)*data_split[0])]
        dev_data = data[int(len(data)*data_split[0]):int(len(data)*(data_split[0]+data_split[1]))]
        test_data = data[int(len(data)*(data_split[0]+data_split[1])):]
    elif experiment == "common_gen":
        train_data = data["train"]
        dev_data = data["val"]
        train_data = [ x for x in train_data if x["response"].strip() != ""]
        dev_data = [ x for x in dev_data if x["response"].strip() != ""]
        test_data = dev_data[400:]
        dev_data = dev_data[:400]
    elif experiment == "MI_rl":
        train_data = data["train"][:1000]
        dev_data = data["val"]
        test_data = data["test"] + data["train"][1000:]
    elif experiment == "webnlg":
        train_data = data["train"]
        dev_data = data["val"]
        test_data = data["test"][:400] 
    else:
        train_data = data["train"]
        dev_data = data["val"]
        test_data = data["test"]
    print("="*30)
    print("Train data size:", len(train_data))
    print("Dev data size:", len(dev_data))
    print("Test data size:", len(test_data))
    print("="*30)
    if rl_validation_size != -1:
        rl_validation_size = min(len(dev_data), rl_validation_size)
        print("rl_validation_size:", rl_validation_size)
        dev_data = random.sample(dev_data, rl_validation_size)
    DEBUG = debug
    if DEBUG:
        train_data = train_data[:100]
        test_data = test_data[:100]
    return train_data, dev_data, test_data
def get_scorers(
    experiment: str,
    learning_mode: str,
    single_reward_idx: int = 0,
    train: bool = True
    ):
    if experiment.endswith("empathy_EX_ER"):
        scorers = [
            {"name": "Empathy_Exploration", "model": Empathy(type="EX"), "sign": 1, "weight": 1.0, "train": True},
            {"name": "Empathy_EmotionalReaction", "model": Empathy(type="ER"), "sign": 1, "weight": 1.0, "train": True},
        ]
    elif experiment.endswith("MI_rl") or experiment.endswith("MI"):
        scorer_model = ReflectionScoreDeployedCL(score_change=False, model_file= "./weights/reflection_scorer_weight.pt")
        scorer_model.type = "CLM"
        scorers = [
            {"name": "reflection_cl", "model": scorer_model, "sign": 1, "weight": 1.0, "train": True},
            {"name": "perplexity_rl", "model": Multi(type="perplexity_rl"), "sign": 1, "weight": 1.0, "train": True},
            {"name": "coherence", "model": Multi(type="coherence", experiment="MI_rl"), "sign": 1, "weight": 1.0, "train": True},
            {"name": "edit_rate", "model": Multi(type="edit_rate"), "sign": 1, "weight": 1.0, "train": False},
            {"name": "diversity-2", "model": Multi(type="diversity-2"), "sign": 1, "weight": 1.0, "train": False },
        ]
    elif experiment.endswith("cnn_daily"):
        scorers = [
            {"name": "summary_rouge", "model": Summary(type="rouge"), "sign": 1, "weight": 1.0, "train": True},
            {"name": "summary_bleu", "model": Summary(type="bleu"), "sign": 1, "weight": 1.0, "train": True},
        ]
    elif experiment.endswith("common_gen"):
        scorers = [
            {"name": "cgen_rouge", "model": Summary(type="rouge"), "sign": 1, "weight": 1.0, "train": True},
            {"name": "cgen_meteor", "model": Summary(type="meteor"), "sign": 1, "weight": 1.0, "train": True},
        ]
    elif experiment.endswith("empathy_full"):
        scorers = [
            {"name": "Empathy_Exploration", "model": Empathy(type="EX"), "sign": 1, "weight": 1.0, "train": True},
            {"name": "Empathy_EmotionalReaction", "model": Empathy(type="ER"), "sign": 1, "weight": 1.0, "train": True},
            {"name": "Empathy_Interpretation", "model": Empathy(type="IP"), "sign": 1, "weight": 1.0, "train": True},
        ]
    elif experiment.endswith("webnlg"):
        scorers = [
            {"name": "WebNLG_BLEU", "model": WebNLG(type="bleu"), "sign": 1, "weight": 1.0, "train": True},
            {"name": "WebNLG_RougeL", "model": WebNLG(type="rougeL"), "sign": 1, "weight": 1.0, "train": True},
            {"name": "WebNLG_Entailment", "model": WebNLG(type="entailment"), "sign": 1, "weight": 1.0, "train": True},
            {"name": "WebNLG_TER", "model": WebNLG(type="ter"), "sign": 1, "weight": 1.0, "train": False},
            {"name": "WebNLG_METEOR", "model": WebNLG(type="meteor"), "sign": 1, "weight": 1.0, "train": False},
        ]
    if learning_mode == "single":
        scorers = [ scorers[single_reward_idx] ]
    if train:
        scorers = [ x for x in scorers if x["train"] ]
    return scorers
def get_naturalness_scorers(
    learning_mode: str,
    single_reward_idx: int = 0,
    ):    
    scorers = [
        {"name": "perplexity", "model": Multi(type="perplexity"), "sign": 1, "weight": 1.0},
        {"name": "coherence", "model": Multi(type="coherence"), "sign": 1, "weight": 1.0},
        {"name": "specificity", "model": Multi(type="specificity"), "sign": 1, "weight": 1.0},
        {"name": "diversity-1", "model": Multi(type="diversity-1"), "sign": 1, "weight": 1.0},
        {"name": "diversity-2", "model": Multi(type="diversity-2"), "sign": 1, "weight": 1.0},
    ]
    if learning_mode == "single":
        scorers = [ scorers[single_reward_idx] ]
    return scorers
