import utils_misc, time, argparse, numpy as np, wandb
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig, BartModel
from transformers import AutoModel, AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration, T5Config, AutoModelForSeq2SeqLM
from datasets import load_dataset
from datetime import datetime
import torch
from tqdm import tqdm
import torch.nn.functional as F
import os
import json
from typing import Optional, List, Dict, Tuple
import fire
import utils_optim
from process_data import *
from peft_util import *
from random_util import set_seed
from test_util import *
from experiment_util import *
from nltk import sent_tokenize
def run_supervised_train(
    model_name: str = "t5-base",
    model_start_dir: Optional[str] = None,
    max_seq_length: int = 90,
    learning_rate: float = 1e-4,
    optimizer_name: str = "adam",
    train_batch_size: int = 8,
    num_epochs: int = 1,
    use_apex: bool = True,
    experiment = "empathy_EX_ER",
    learning_mode = "bandit_weighted",
    grad_accum_steps = 1,
    data_list: Optional[List] = None,
    lora=False,
    load_in_8bit=False,
    test=True,
    debug=False,
    seed = 420691488,
    ):
    set_seed(seed)
    data_split = [0.8, 0.1, 0.1]
    train_data, dev_data, test_data = get_data(experiment, data_split, -1, debug, supervised=True)
    gen_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    wandb.init(project="scst")
    experiment = "supervised_"+experiment
    wandb.run.name = experiment
    wandb.run.save()
    wandb.config.update({"model_name": model_name, 
        "learning_rate": learning_rate, 
        "optimizer_name": optimizer_name, 
        "train_batch_size": train_batch_size, 
        "num_epochs": num_epochs, 
        "use_apex": use_apex, 
        "experiment": experiment, 
        "learning_mode": learning_mode,
        "grad_accum_steps": grad_accum_steps, 
        "data_list": data_list, 
        "lora": lora,
        "load_in_8bit": load_in_8bit, 
        "test": test, 
        "debug": debug,
        "seed": seed,}
    )
    begin_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output_dir =  "./webnlg_outputs/%s_%s/" % (experiment, begin_time)
    if experiment == "cnn_daily":
        max_seq_length = 512
    model, tokenizer = get_model(model_name, model_start_dir, max_seq_length=max_seq_length, lora=lora)
    model.to(gen_device)
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
    scaler = torch.cuda.amp.GradScaler()
    step_count = 0
    for epoch in range(num_epochs):
        for paragraphs in (pbar := tqdm(dataloader, position=0, leave=True, dynamic_ncols=True)):
            responses = paragraphs["responses"]
            prompts = paragraphs["prompts"]
            """
            special segment begin
            """
            sents = [sent_tokenize(g) for g in responses]
            sents = [ [s + " [CLS]" for s in sent] for sent in sents]
            responses = [ " ".join(sent) for sent in sents]
            """
            special segment end
            """
            with torch.cuda.amp.autocast():
                encoded_responses = tokenizer(responses, padding="longest", truncation=True, return_tensors="pt")
                encoded_prompts = tokenizer(prompts, padding="longest", truncation=True, return_tensors="pt")
                encoded_prompts = encoded_prompts.to(gen_device)
                encoded_responses = encoded_responses.to(gen_device)
                output = model(**encoded_prompts, labels=encoded_responses["input_ids"])
                loss = output.loss / grad_accum_steps
                pbar.set_description(f"Loss: {loss.item():.2f}")
                wandb.log({"loss": loss.item()})
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
            step_count += 1
    model_file_name= model_name.replace("/", "_")
    save_name =  f"{experiment}_epochs{num_epochs}"
    model.save_pretrained(f"{output_dir}/{save_name}")
    tokenizer.save_pretrained(f"{output_dir}/{save_name}")
    if test:
        run_test(model, tokenizer, test_data, test_batch_size=16, experiment=experiment, output_dir=output_dir)
    return model, tokenizer
if __name__ == '__main__':
    fire.Fire(run_supervised_train)