import torch, time, numpy as np
import utils_misc
from bandit_alg import Exp3
class ScorerWrapper:
    def __init__(self, scorers, learning_mode = "single", bandit = None, scoring_method="logsum",\
     max_batch_size=100, use_caching=False, rl_validation_step=20, total_step_count=1000):
        assert scoring_method in ["sum", "product", "logsum"], "Unrecognized `scoring_method`"
        self.scorers = scorers
        self.scoring_method = scoring_method
        self.use_caching = use_caching
        self.cache = {}
        self.rl_validation_step = rl_validation_step
        self.max_batch_size = max_batch_size
        if self.scoring_method == "logsum":
            self.score_func = logsum_score
        elif self.scoring_method == "product":
            self.score_func = product_score
        elif self.scoring_method == "sum":
            self.score_func = sum_score
        self.learning_mode = learning_mode
        self.used_scorers = []
        if self.learning_mode == "bandit_weighted" or self.learning_mode == "argmin" or self.learning_mode == "contextual":
            self.weight_bandit = Exp3(len(self.scorers))
        self.total_step_count = total_step_count
    def get_score_names(self):
        return [s["name"] for s in self.scorers]
    def make_key(self, inp, gen):
        return "%s|||___|||%s" % (inp, gen)
    def score(self, inputs, generateds, partial=False, printing=False, timings=False, \
        extras={}, progress=False, responses=None, step_count=None, bandit=None, chosen=None):
        assert len(inputs) == len(generateds), "Input and output lengths don't match"
        if self.learning_mode in ["bandit", "bandit_weighted"] and bandit is None:
            raise Exception("Bandit is not defined")
        if self.learning_mode == "single" or self.learning_mode == "weighted":
            self.used_scorers = self.scorers
        elif self.learning_mode == "bandit":
            if chosen != None:
                self.used_scorers = [self.scorers[chosen]]
                print("Used scorers:", self.used_scorers)
        elif self.learning_mode == "bandit_weighted":
            self.used_scorers = self.scorers
            if chosen != None:
                if int(chosen) == len(self.scorers):
                    pass
                else:
                    self.weight_bandit(1, chosen)
                    print("Weight Bandit:", self.weight_bandit.weights)
                    weights = self.weight_bandit.weights
                    weights = weights / np.sum(weights)
                    self.used_scorers = self.scorers
                    for i, scorer in enumerate(self.used_scorers):
                        scorer["weight"] = weights[i]
                    print([scorer["weight"] for scorer in self.used_scorers])
        elif self.learning_mode == "argmin":
            if chosen != None:
                self.weight_bandit(1, chosen)
                print("Weight Bandit:", self.weight_bandit.weights)
                weights = self.weight_bandit.weights
                weights = weights / np.sum(weights)
                self.used_scorers = self.scorers
                for i, scorer in enumerate(self.used_scorers):
                    scorer["weight"] = weights[i]
                print([scorer["weight"] for scorer in self.used_scorers])
        elif self.learning_mode == "bayes":
            if chosen != None:
                chosen  = [c / np.sum(chosen) for c in chosen]
                self.used_scorers = self.scorers
                for i, scorer in enumerate(self.used_scorers):
                    scorer["weight"] = chosen[i]
                print("Bayes set weights")
                print([scorer["weight"] for scorer in self.used_scorers])
        elif self.learning_mode == "contextual":
            self.used_scorers = self.scorers
            if chosen != None:
                if int(chosen[0]) == len(self.scorers):
                    pass
                else:
                    self.weight_bandit(1, int(chosen[0]))
                    print("Weight Bandit:", self.weight_bandit.weights)
                    weights = self.weight_bandit.weights
                    weights = weights / np.sum(weights)
                    self.used_scorers = self.scorers
                    for i, scorer in enumerate(self.used_scorers):
                        scorer["weight"] = weights[i]
                    print([scorer["weight"] for scorer in self.used_scorers])
        elif self.learning_mode == "round":
            self.used_scorers = self.scorers
            interval = int(self.total_step_count / len(self.scorers))
            chosen = step_count // interval
            chosen = chosen % len(self.scorers)
            print("Chosen:", chosen)
            self.used_scorers = [self.scorers[chosen]]
            print("Used scorers:", self.used_scorers)
        else:
            self.used_scorers = self.scorers
        if not self.use_caching:
            self.cache = {} 
        todo = []
        all_keys = []
        for inp, gen, response in zip(inputs, generateds, responses):
            key = self.make_key(inp, gen)
            all_keys.append(key)
            if key not in self.cache:
                todo.append({"inp": inp, "gen": gen, "key": key, "response": response})
        for d in todo:
            self.cache[d["key"]] = {}
        if self.use_caching and len(todo) < len(all_keys):
            print("With caching, only processing: %d / %d samples" % (len(todo), len(all_keys)))
        if len(todo) == 0:
            progress = False 
        for batch_todo in utils_misc.batcher(todo, batch_size=self.max_batch_size, progress=progress):
            batch_inputs = [d["inp"] for d in batch_todo]
            batch_gens = [d["gen"] for d in batch_todo]
            batch_responses = [d["response"] for d in batch_todo]
            batch_scores, timings_out = self.score_func(self.used_scorers, batch_inputs, batch_gens,\
                 partial=partial, printing=printing, extras=extras, responses=batch_responses)
            for k, out in batch_scores.items():
                if type(out) in [torch.Tensor, np.array, np.ndarray]:
                    out = out.tolist()
                for i, d in enumerate(batch_todo):
                    self.cache[d["key"]][k] = out[i]
            if timings:
                print(timings_out)
        all_outputs = {}
        for k in self.cache[all_keys[0]].keys():
            all_outputs[k] = [self.cache[key][k] for key in all_keys]
        if printing:
            print("[total]", all_outputs["total_scores"])
        return all_outputs
    def rl_score(self, inputs, generateds, partial=False, printing=False, timings=False, \
        extras={}, progress=False, responses=None, step_count=None, bandit=None, chosen=None):
        assert len(inputs) == len(generateds), "Input and output lengths don't match"
        if not self.use_caching:
            self.cache = {} 
        todo = []
        all_keys = []
        for inp, gen, response in zip(inputs, generateds, responses):
            key = self.make_key(inp, gen)
            all_keys.append(key)
            if key not in self.cache:
                todo.append({"inp": inp, "gen": gen, "key": key, "response": response})
        for d in todo:
            self.cache[d["key"]] = {}
        if self.use_caching and len(todo) < len(all_keys):
            print("With caching, only processing: %d / %d samples" % (len(todo), len(all_keys)))
        if len(todo) == 0:
            progress = False 
        for batch_todo in utils_misc.batcher(todo, batch_size=self.max_batch_size, progress=progress):
            batch_inputs = [d["inp"] for d in batch_todo]
            batch_gens = [d["gen"] for d in batch_todo]
            batch_responses = [d["response"] for d in batch_todo]
            batch_scores, timings_out = self.score_func(self.scorers, batch_inputs, batch_gens,\
                 partial=partial, printing=printing, extras=extras, responses=batch_responses)
            for k, out in batch_scores.items():
                if type(out) in [torch.Tensor, np.array, np.ndarray]:
                    out = out.tolist()
                for i, d in enumerate(batch_todo):
                    self.cache[d["key"]][k] = out[i]
            if timings:
                print(timings_out)
        all_outputs = {}
        for k in self.cache[all_keys[0]].keys():
            all_outputs[k] = [self.cache[key][k] for key in all_keys]
        if printing:
            print("[total]", all_outputs["total_scores"])
        return all_outputs
    def __call__(self, inputs, generateds, **kwargs):
        return self.score(inputs, generateds, **kwargs)
def sum_score(scorers, paragraphs, generateds, partial=False, printing=False, extras={}):
    total_scores = np.zeros((len(paragraphs)))
    scorer_returns, timings = {}, {}
    T = time.time()
    for scorer in scorers:
        scores = scorer['model'].score(paragraphs, generateds, partial=partial, printing=printing, **extras)
        weight = scorer.get("weight", 1.0)
        total_scores += scorer["sign"]*weight*np.array(scores['scores'])
        scorer_returns.update({scorer['name']+"_"+k: v for k, v in scores.items()})
        timings[scorer["name"]] = time.time()-T
        T = time.time()
    scorer_returns['total_scores'] = total_scores
    return scorer_returns, timings
def product_score(scorers, paragraphs, generateds, partial=False, printing=False, extras={}):
    total_scores = np.ones((len(paragraphs)))
    scorer_returns, timings = {}, {}
    T = time.time()
    for scorer in scorers:
        scores = scorer['model'].score(paragraphs, generateds, partial=partial, printing=printing, **extras)
        if scorer['sign'] == 1:
            total_scores *= np.array(scores['scores'])
        else: 
            total_scores *= (1-np.array(scores['scores']))
        scorer_returns.update({scorer['name']+"_"+k: v for k, v in scores.items()})
        timings[scorer["name"]] = time.time()-T
        T = time.time()
    scorer_returns['total_scores'] = total_scores
    return scorer_returns, timings
def logsum_score(scorers, paragraphs, generateds, partial=False, printing=False, \
    extras={}, reflection_labels=None, responses=None):
    total_scores = np.zeros((len(paragraphs)))
    raw_total_scores = np.zeros((len(paragraphs)))
    scorer_returns, timings = {}, {}
    T = time.time()
    for scorer in scorers:
        if "disc" in scorer["name"] or "summary" in scorer["name"] or "cgen" in scorer["name"]:
            scores = scorer['model'].score(paragraphs, generateds, partial=partial, printing=printing, \
                reflection_labels=reflection_labels, responses=responses, **extras)
        else:
            scores = scorer['model'].score(paragraphs, generateds, partial=partial, printing=printing, responses = responses,**extras)
        weight = scorer.get("weight", 1.0)
        if not scorer["name"].endswith("perplexity"):
            scores["scores"] = np.clip(scores["scores"], 0.0001, 0.9999)
        """
        if "reflection" not in scorer["name"]:
            scores["scores"] = np.clip(scores["scores"], 0.0001, 0.9999)
        if "reflection" in scorer["name"]:
            total_scores += weight*np.log(1+np.array(scores['scores']))
        """
        if scorer['sign'] == 1:
            total_scores += weight*np.log(np.array(scores['scores']))
            raw_total_scores += np.log(np.array(scores['scores']))
        else: 
            total_scores += weight * np.log(1-np.array(scores["scores"]))
            raw_total_scores += np.log(1-np.array(scores["scores"]))
        scorer_returns.update({scorer['name']+"_"+k: v for k, v in scores.items()})
        timings[scorer["name"]] = time.time()-T
        T = time.time()
    scorer_returns['total_scores'] = total_scores
    scorer_returns['raw_total_scores'] = raw_total_scores
    return scorer_returns, timings
