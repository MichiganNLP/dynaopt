import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, AutoModelForMaskedLM
from transformers import BertTokenizer, BertForSequenceClassification, BertModel, BertForMaskedLM
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import utils_optim
from tqdm import tqdm
import fire
import transformers
transformers.logging.set_verbosity_error()
def score_coherence(prompt, response, model, tokenizer):
    encoded_prompt_response = tokenizer(prompt, response, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model(**encoded_prompt_response)
        logits = outputs.logits
    score = logits.softmax(dim=-1)[:,-1].tolist()
    coherence = score
    return coherence
import random
def create_coherence_data(data):
    new_data = []
    for d in tqdm(data):
        prompt = d["prompt"]
        response = d["response"]
        while True:
            random_picked = random.choice(data)
            picked_prompt  = random_picked["prompt"]
            anti_response = random_picked["response"]
            if anti_response != response and picked_prompt != prompt:
                break
        pos_pair = {"prompt": prompt, "response": response, "label": 1}
        neg_pair = {"prompt": prompt, "response": anti_response, "label": 0}
        new_data.append(pos_pair)
        new_data.append(neg_pair)
    return new_data
from experiment_util import *
def train_coherence(
        experiment: str = 'MI_rl',
        train_batch_size: int = 8,
        epochs=1,
        lr=1e-5,
    ):
    try:
        with open(f"data/{experiment}_coherence_data.json", "r") as f:
            data = json.load(f)
        train_data = data["train"]
        dev_data = data["dev"]
        test_data = data["test"]
    except:
        data_split = [0.8, 0.1, 0.1]
        train_data, dev_data, test_data = get_data(experiment, data_split, -1, False)
        train_data = create_coherence_data(train_data)
        dev_data = create_coherence_data(dev_data)
        test_data = create_coherence_data(test_data)
        with open(f"data/{experiment}_coherence_data.json", "w") as f:
            json.dump({"train": train_data, "dev": dev_data, "test": test_data}, f, indent=4)
    if True:
        train_data = train_data[:1000]
    print("coherence train data size:", len(train_data))
    print("coherence dev data size:", len(dev_data))
    print("coherence test data size:", len(test_data))
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    if torch.cuda.device_count() > 1:
        device = torch.device('cuda:1')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = utils_optim.build_optimizer(model, optimizer_name="adamw", learning_rate=lr)
    def batch_collate(inps):
        batch_paras = []
        batch_labels = []
        batch_responses = [] 
        for inp in inps:
            text = inp["prompt"] 
            batch_paras.append(text)
            batch_responses.append(inp["response"])
            batch_labels.append(inp["label"])
        return {"prompts": batch_paras,
                "responses": batch_responses,
                "labels": batch_labels
                }
    dataloader = DataLoader(dataset=train_data, batch_size=train_batch_size,\
        sampler=RandomSampler(train_data), drop_last=True, collate_fn=batch_collate)
    test_dataloader = DataLoader(dataset=test_data, batch_size=train_batch_size,\
        sampler=SequentialSampler(test_data), drop_last=True, collate_fn=batch_collate)
    model.train()
    for epoch in range(epochs):
        print("Epoch: ", epoch)
        for paragraphs in (pbar := tqdm(dataloader, position=0, leave=True, dynamic_ncols=True)):
            responses = paragraphs["responses"]
            prompts = paragraphs["prompts"]
            labels = paragraphs["labels"]
            encoded_prompt_responses = tokenizer(prompts, responses, return_tensors='pt', padding=True, truncation="longest_first")
            encoded_prompt_responses = encoded_prompt_responses.to(device)
            labels = torch.LongTensor(labels).to(device)
            output = model(**encoded_prompt_responses, labels=labels)
            pred = output.logits.argmax(dim=-1)
            if False:
                for r, p, pr, l in zip(responses, prompts, pred, labels):
                    print("=====================================")
                    print("prompt: ", p)
                    print("response: ", r)
                    print("pred: ", pr)
                    print("label: ", l)
                    print()
            loss = output.loss
            pbar.set_description(f"Loss: {loss.item():.2f}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    truths, predicts = [], []
    model.eval()
    for paragraphs in (pbar := tqdm(test_dataloader, position=0, leave=True, dynamic_ncols=True)):
        responses = paragraphs["responses"]
        prompts = paragraphs["prompts"]
        labels = paragraphs["labels"]
        encoded_prompt_responses = tokenizer(prompts, responses, return_tensors='pt', padding=True, truncation="longest_first")
        encoded_prompt_responses = encoded_prompt_responses.to(device)
        labels = torch.LongTensor(labels).to(device)
        output = model(**encoded_prompt_responses, labels=labels)
        logits = output.logits
        gt = labels.tolist()
        preds = logits.argmax(dim=-1).tolist()
        pbar.set_description(f"Accuracy: {sum([1 if t==p else 0 for t,p in zip(gt, preds)])/len(gt):.2f}")
        truths.extend(gt)
        predicts.extend(preds)
    print("Accuracy: ", sum([1 if t==p else 0 for t,p in zip(truths, predicts)])/len(truths))
    model.save_pretrained(f"models/{experiment}/coherence")
if __name__ == "__main__":
    fire.Fire(train_coherence)