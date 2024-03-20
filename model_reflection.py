from transformers import GPT2LMHeadModel, GPT2TokenizerFast, RobertaForSequenceClassification, RobertaTokenizerFast
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
import numpy as np, tqdm, json, collections, torch
from sklearn.metrics import f1_score
from torch.cuda.amp import autocast
from collections import Counter
import utils_optim
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, AutoModel, AutoModelForMaskedLM
from cross_scorer_model import CrossScorer, CrossScorerCrossEncoder
class ReflectionScoreDeployedCL:
    """
    Uses the cross scorer CL model (= CrossScorerCrossEncoder)
    """
    def __init__(self, same_length=False, score_change=False, model_file="./weights/reflection_scorer_weight.pt"):
        self.same_length = same_length
        model_name = "roberta-base"
        self.encoder = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = CrossScorerCrossEncoder(self.encoder)
        c_ckpt = torch.load(model_file) 
        self.model.load_state_dict(c_ckpt) 
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
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
        max_output_length = 160 
        c_prompts_u_responses = self.tokenizer(c_prompts, u_responses, padding='longest',truncation=True,return_tensors='pt')
        c_prompts_model_responses = self.tokenizer(c_prompts, decoded, padding='longest',truncation=True,return_tensors='pt')
        c_prompts_u_responses = c_prompts_u_responses.to(self.device)
        c_prompts_model_responses = c_prompts_model_responses.to(self.device)
        return c_prompts_u_responses, c_prompts_model_responses
    def sim_func(self, a,b):
        return (self.cos_sim(a,b)+1.0)/2.0
    def score_relative(self, sources, generateds, partial=False, printing=False, **kwargs):
        up_to_length = None
        if self.same_length or partial:
            up_to_length = len(self.tokenizer.encode(generateds[0]))
        c_prompts_u_responses, c_prompts_model_responses = self.preprocess_batch(sources, generateds)
        with torch.no_grad():
            score_pm = self.model.score_forward(**c_prompts_model_responses).sigmoid().squeeze()
            score_pu = self.model.score_forward(**c_prompts_u_responses).sigmoid().squeeze()
        scores = score_pm - score_pu 
        scores = scores.tolist()
        if printing:
            print("[reflection_change]", scores)
        return {"scores": scores  } 
    def score_absolute(self, sources, generateds, partial=False, printing=False, **kwargs):
        up_to_length = None
        if self.same_length or partial:
            up_to_length = len(self.tokenizer.encode(generateds[0]))
        c_prompts_u_responses, c_prompts_model_responses = self.preprocess_batch(sources, generateds)
        with torch.no_grad():
            score_pm, attentions = self.model.score_forward(**c_prompts_model_responses, \
                output_attentions=True, return_attentions=True)
        scores = score_pm
        scores = scores.tolist()
        if printing:
            print("[reflection]", scores)
        return {"scores": scores } 
