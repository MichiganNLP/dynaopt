from transformers import GPT2LMHeadModel, GPT2Tokenizer, RobertaForSequenceClassification, RobertaTokenizerFast
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
import numpy as np, tqdm, json, collections, torch
from sklearn.metrics import f1_score
from torch.cuda.amp import autocast
from collections import Counter
import utils_optim
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, AutoModel, AutoModelForMaskedLM
from transformers import BertModel, BertTokenizer, BertForMaskedLM, BertForSequenceClassification
import os
from itertools import chain
__all__ = ["ngrams"]
def pad_sequence(sequence, n, pad_left=False, pad_right=False,
                 left_pad_symbol=None, right_pad_symbol=None):
    """
    Returns a padded sequence of items before ngram extraction.
        >>> list(pad_sequence([1,2,3,4,5], 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
        ['<s>', 1, 2, 3, 4, 5, '</s>']
        >>> list(pad_sequence([1,2,3,4,5], 2, pad_left=True, left_pad_symbol='<s>'))
        ['<s>', 1, 2, 3, 4, 5]
        >>> list(pad_sequence([1,2,3,4,5], 2, pad_right=True, right_pad_symbol='</s>'))
        [1, 2, 3, 4, 5, '</s>']
    :param sequence: the source data to be padded
    :type sequence: sequence or iter
    :param n: the degree of the ngrams
    :type n: int
    :param pad_left: whether the ngrams should be left-padded
    :type pad_left: bool
    :param pad_right: whether the ngrams should be right-padded
    :type pad_right: bool
    :param left_pad_symbol: the symbol to use for left padding (default is None)
    :type left_pad_symbol: any
    :param right_pad_symbol: the symbol to use for right padding (default is None)
    :type right_pad_symbol: any
    :rtype: sequence or iter
    """
    sequence = iter(sequence)
    if pad_left:
        sequence = chain((left_pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = chain(sequence, (right_pad_symbol,) * (n - 1))
    return sequence
def ngrams(sequence, n, pad_left=False, pad_right=False,
           left_pad_symbol=None, right_pad_symbol=None):
    """
    Return the ngrams generated from a sequence of items, as an iterator.
    For example:
        >>> from nltk.util import ngrams
        >>> list(ngrams([1,2,3,4,5], 3))
        [(1, 2, 3), (2, 3, 4), (3, 4, 5)]
    Wrap with list for a list version of this function.  Set pad_left
    or pad_right to true in order to get additional ngrams:
        >>> list(ngrams([1,2,3,4,5], 2, pad_right=True))
        [(1, 2), (2, 3), (3, 4), (4, 5), (5, None)]
        >>> list(ngrams([1,2,3,4,5], 2, pad_right=True, right_pad_symbol='</s>'))
        [(1, 2), (2, 3), (3, 4), (4, 5), (5, '</s>')]
        >>> list(ngrams([1,2,3,4,5], 2, pad_left=True, left_pad_symbol='<s>'))
        [('<s>', 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        >>> list(ngrams([1,2,3,4,5], 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
        [('<s>', 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, '</s>')]
    :param sequence: the source data to be converted into ngrams
    :type sequence: sequence or iter
    :param n: the degree of the ngrams
    :type n: int
    :param pad_left: whether the ngrams should be left-padded
    :type pad_left: bool
    :param pad_right: whether the ngrams should be right-padded
    :type pad_right: bool
    :param left_pad_symbol: the symbol to use for left padding (default is None)
    :type left_pad_symbol: any
    :param right_pad_symbol: the symbol to use for right padding (default is None)
    :type right_pad_symbol: any
    :rtype: sequence or iter
    """
    sequence = pad_sequence(sequence, n, pad_left, pad_right,
                            left_pad_symbol, right_pad_symbol)
    history = []
    while n > 1:
        history.append(next(sequence))
        n -= 1
    for item in sequence:
        history.append(item)
        yield tuple(history)
        del history[0]
__all__ = ["distinct_n_sentence_level", "distinct_n_corpus_level"]
def distinct_n_sentence_level(sentence, n):
    """
    Compute distinct-N for a single sentence.
    :param sentence: a list of words.
    :param n: int, ngram.
    :return: float, the metric value.
    """
    if len(sentence) == 0:
        return 0.0  
    distinct_ngrams = set(ngrams(sentence, n))
    return len(distinct_ngrams) / len(sentence)
def distinct_n_corpus_level(sentences, n):
    """
    Compute average distinct-N of a list of sentences (the corpus).
    :param sentences: a list of sentence.
    :param n: int, ngram.
    :return: float, the average value.
    """
    return sum(distinct_n_sentence_level(sentence, n) for sentence in sentences) / len(sentences)
def single_score_perplexity(prompts, responses, model, tokenizer):
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.max_length = 512
    inputs = tokenizer(prompts, responses, add_special_tokens=True, padding=True, truncation=True, return_tensors='pt').to(model.device)
    labels = inputs['input_ids']
    outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=labels, return_dict=True)
    loss = outputs.loss
    num_tokens = torch.sum(inputs['attention_mask']).item()
    perplexity = torch.exp(loss).item()
    return perplexity
def score_perplexity(prompts, responses, model, tokenizer):
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.max_length = 512
    perplexities = [ single_score_perplexity([prompt], [response], model, tokenizer) for prompt, response in zip(prompts, responses)    ]
    return perplexities
def score_specificity(prompt, response, model, tokenizer):
    encoded_prompt = tokenizer(prompt, return_tensors='pt', padding="longest", truncation=True).to(model.device)
    encoded_response = tokenizer(response, return_tensors='pt', padding="longest", truncation=True).to(model.device)
    with torch.no_grad():
        prompt_embeddings = model(**encoded_prompt).last_hidden_state[:, 0, :]
        response_embeddings = model(**encoded_response).last_hidden_state[:, 0, :]
    similarity = torch.cosine_similarity(prompt_embeddings, response_embeddings).tolist()
    return similarity
def score_coherence(prompt, response, model, tokenizer):
    encoded_prompt_response = tokenizer(prompt, response, return_tensors='pt', truncation=True, padding="longest").to(model.device)
    with torch.no_grad():
        outputs = model(**encoded_prompt_response)
        logits = outputs.logits
    score = logits.softmax(dim=-1)[:,-1].tolist()
    coherence = score
    return coherence
from functools import partial
from utils_edits import build_levenshtein_matrix
class Multi:
    """
    Uses the cross scorer CL model (= CrossScorerCrossEncoder)
    """
    def __init__(self, same_length=False, score_change=False, type="perplexity", batch_size=32, tokenizer=None, experiment="empathy_full"):
        self.same_length = same_length
        self.type = type
        self.batch_size = batch_size 
        if torch.cuda.device_count() > 1:
            self.device = torch.device('cuda:1')
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.experiment = experiment
        if self.type =="perplexity":
            self.model = GPT2LMHeadModel.from_pretrained('gpt2')
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.model.to(self.device)
            self.metric = partial(score_perplexity, model=self.model, tokenizer=self.tokenizer)
        elif self.type =="perplexity_rl":
            self.model = GPT2LMHeadModel.from_pretrained('gpt2')
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.model.to(self.device)
            metric = partial(score_perplexity, model=self.model, tokenizer=self.tokenizer)
            def new_metric(x, y):
                CAP = 50.0
                result = metric(x, y)
                return [(CAP-r)/CAP for r in result]
            self.metric = new_metric
        elif self.type == "coherence":
            self.model = BertForSequenceClassification.from_pretrained(f'models/{experiment}/coherence')
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model.to(self.device)
            self.metric = partial(score_coherence, model=self.model, tokenizer=self.tokenizer)
        elif self.type == "specificity":
            self.model = BertModel.from_pretrained('bert-base-uncased')
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model.to(self.device)
            self.metric = partial(score_specificity, model=self.model, tokenizer=self.tokenizer)
        elif self.type == "diversity-1":
            def div1(prompts, generateds):
                res = []
                for ss in generateds:
                    res += [distinct_n_sentence_level(ss.split(), 1)]
                return res
            self.metric = div1
        elif self.type == "diversity-2":
            def div2(prompts, generateds):
                res = []
                for ss in generateds:
                    res += [distinct_n_sentence_level(ss.split(), 2)]
                return res
            self.metric = div2
        elif self.type == "edit_rate":
            def edit_rate(prompts, generateds):
                res = []
                for pp, ss in zip(prompts, generateds):
                    lev_mat, u,m = build_levenshtein_matrix(pp, ss)
                    res += [ (float(lev_mat[-1][-1]))/max(len(u),len(m)) if max(len(u), len(m)) > 0 else 0 ]
                return res
            self.metric = edit_rate
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
        raise NotImplementedError
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
        c_prompts, u_responses, model_responses = self.preprocess_batch(sources, generateds)
        with torch.no_grad():
            score_pm = self.metric(c_prompts, model_responses)
        scores = score_pm
        if printing:
            print("[reflection]", scores)
        return {"scores": scores }
def main():
    prompts = ["Hello, how are you? [SEP]", "What's your favorite color? [SEP]", "I am tired of being alone. [SEP]"]
    responses = ["I'm doing well, thanks for asking.", "My favorite color is blue.", "Have you tried this hamburger? It's delicious."]
    if False:
        perplexity = Multi(type="perplexity")
        a = perplexity.score(prompts, responses)
        print(a)
    if True:
        coherence = Multi(type="coherence")
        a = coherence.score(prompts, responses)
        print(a)
    if False:
        specificity = Multi(type="specificity")
        a = specificity.score(prompts, responses)
        print(a)
    if False:
        diversity = Multi(type="diversity-1")
        a = diversity.score(prompts, responses)
        print(a)
    if False:
        diversity = Multi(type="diversity-2")
        a = diversity.score(prompts, responses)
        print(a)
if __name__ == "__main__":
    main()