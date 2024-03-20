from transformers import GPT2LMHeadModel, GPT2TokenizerFast, RobertaForSequenceClassification, RobertaTokenizerFast
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
import numpy as np, tqdm, json, collections, torch
from sklearn.metrics import f1_score
from torch.cuda.amp import autocast
from collections import Counter
import utils_optim
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, AutoModel, AutoModelForMaskedLM
import os
import evaluate
from nltk.translate import meteor
from nltk import word_tokenize
import nltk
def compute_meteor(prediction, reference):
    result = nltk.translate.meteor_score.meteor_score([word_tokenize(reference)], word_tokenize(prediction))
    return result
def batch_meteor(predictions, references):
    results = []
    for pred, ref in zip(predictions, references):
        try:
            results.append(compute_meteor(pred, ref))
        except:
            results.append(0.0)
    return results
class Summary:
    """
    Uses the cross scorer CL model (= CrossScorerCrossEncoder)
    """
    def __init__(self, same_length=False, score_change=False, type="rouge", batch_size=32):
        self.same_length = same_length
        self.type = type
        self.batch_size = batch_size 
        if self.type == "meteor":
            print("Loading meteor")
            self.metric = batch_meteor
        elif self.type == "bleu":
            print("Loading bleu")
            metric = evaluate.load(self.type)
            def compute(x,y):
                scores = []
                for xx,yy in zip(x,y):
                    try:
                        res = metric.compute(predictions=[xx], references=[[yy]])
                    except:
                        res = {"bleu": 0.0}
                    scores.append(res["bleu"])
                return scores
            self.metric = compute 
        elif self.type == "rouge":
            print("Loading rouge")
            metric = evaluate.load(self.type)
            def compute(x,y):   
                scores = []
                for xx,yy in zip(x,y):
                    try:
                        scores.append(metric.compute(predictions=[xx], references=[[yy]])["rougeL"])
                    except:
                        scores.append(0.0)
                return scores
            self.metric = compute 
        self.score_change = score_change
        if self.score_change:
            self.score = self.score_relative
        else:
            self.score = self.score_absolute
    def preprocess_batch(self, sources, decoded):
        c_prompts = []
        u_responses = []
        for source, decod in zip(sources, decoded):
            sp = source.split("[SEP]")
            if len(sp) != 2:
                print("Error, formatting must be wrong")
                print("source:", source)
                continue    
            client_prompt = sp[0].strip()
            c_prompts += [ client_prompt  ]
            u_responses += [ sp[1] ]
        return c_prompts, u_responses, decoded
    def score_relative(self, sources, generateds, partial=False, printing=False, **kwargs):
        up_to_length = None
        c_prompts, u_responses, model_responses = self.preprocess_batch(sources, generateds)
        with torch.no_grad():
            score_pm = self.metric(c_prompts, model_responses)
            score_pu = self.metric(c_prompts, u_responses)
        scores = [ s-p for s,p in zip(score_pu, score_pm)] 
        if printing:
            print("[reflection_change]", scores)
        return {"scores": scores  } 
    def score_absolute(self, sources, generateds, partial=False, printing=False, responses =None, **kwargs):
        up_to_length = None
        if responses is None:
            print("Error!")
            exit()
        c_prompts, u_responses, model_responses = self.preprocess_batch(sources, generateds)
        with torch.no_grad():
            score_pm = self.metric(model_responses, responses)
        scores = score_pm
        if printing:
            print("[reflection]", scores)
        return {"scores": scores }
