"""
Bandit Algorithms
exp3 was written by J Kun: https://github.com/j2kun/exp3/blob/main/exp3.py
"""
import math
import random
def draw(weights):
    if len(weights) == 0:
        return 0
    choice = random.uniform(0, sum(weights))
    choiceIndex = 0
    for weight in weights:
        choice -= weight
        if choice <= 0:
            return choiceIndex
        choiceIndex += 1
def distr(weights, gamma=0.0):
    theSum = float(sum(weights))
    return tuple((1.0 - gamma) * (w / theSum) + (gamma / len(weights)) for w in weights)
def exp3(numActions, reward, gamma, rewardMin = 0, rewardMax = 1):
   weights = [1.0] * numActions
   t = 0
   while True:
      probabilityDistribution = distr(weights, gamma)
      choice = draw(probabilityDistribution)
      theReward = reward(choice, t)
      scaledReward = (theReward - rewardMin) / (rewardMax - rewardMin) 
      estimatedReward = 1.0 * scaledReward / probabilityDistribution[choice]
      weights[choice] *= math.exp(estimatedReward * gamma / numActions) 
      yield choice, theReward, estimatedReward, weights
      t = t + 1
class Exp3:
    def __init__(self, numActions, gamma = 0.07, rewardMin = 0, rewardMax = 1):
        self.numActions = numActions
        self.gamma = gamma
        self.rewardMin = rewardMin
        self.rewardMax = rewardMax
        self.weights = [1.0] * numActions
        self.t = 0
    """
    Assumption here:
        The user first calls draw() to get the choice, then calls __call__() with the reward and choice info
    """
    def __call__(self, reward, choice):
        probabilityDistribution = distr(self.weights, self.gamma)
        theReward = reward
        scaledReward = (theReward - self.rewardMin) / (self.rewardMax - self.rewardMin)
        estimatedReward = 1.0 * scaledReward / probabilityDistribution[choice]
        self.weights[choice] *= math.exp(estimatedReward * self.gamma / self.numActions)
        self.t += 1
        return scaledReward
    def draw(self):
        drawn = draw(distr(self.weights, self.gamma))
        return drawn
    def reset(self):
        self.weights = [1.0] * self.numActions
        self.t = 0
class AlternatingMultiTask:
    def __init__(self, numActions, gamma = 0.07, rewardMin = 0, rewardMax = 1, every=5):
        self.numActions = numActions
        self.gamma = gamma
        self.rewardMin = rewardMin
        self.rewardMax = rewardMax
        self.weights = [1.0] * numActions
        self.t = 0
        self.every = 5
    """
    Assumption here:
        The user first calls draw() to get the choice, then calls __call__() with the reward and choice info
    """
    def __call__(self, reward, choice):
        """
        Don't even call this function
        """
        return None
    def draw(self):
        if self.t % self.every == 0:
            self.drawn += 1
            if self.drawn >= self.numActions:
                self.drawn = 0
        self.t += 1
        drawn = self.drawn
        return drawn
    def reset(self):
        self.weights = [1.0] * self.numActions
        self.t = 0
def simpleTest():
    numActions = 10
    numRounds = 10000
    biases = [1.0 / k for k in range(2,12)]
    rewardVector = [[1 if random.random() < bias else 0 for bias in biases] for _ in range(numRounds)]
    rewards = lambda choice, t: rewardVector[t][choice]
    bestAction = max(range(numActions), key=lambda action: sum([rewardVector[t][action] for t in range(numRounds)]))
    bestUpperBoundEstimate = 2 * numRounds / 3
    gamma = math.sqrt(numActions * math.log(numActions) / ((math.e - 1) * bestUpperBoundEstimate))
    gamma = 0.07
    cumulativeReward = 0
    bestActionCumulativeReward = 0
    weakRegret = 0
    bandit = Exp3(numActions, gamma)
    t = 0
    for rv in rewardVector:
        (choice, reward, est, weights) = bandit(rv)
        cumulativeReward += reward
        bestActionCumulativeReward += rewardVector[t][bestAction]
        weakRegret = (bestActionCumulativeReward - cumulativeReward)
        regretBound = (math.e - 1) * gamma * bestActionCumulativeReward + (numActions * math.log(numActions)) / gamma
        print("regret: %d\tmaxRegret: %.2f\tweights: (%s)" % (weakRegret, regretBound, ', '.join(["%.3f" % weight for weight in distr(weights)])))
        t += 1
        if t >= numRounds:
            break
    print(cumulativeReward)
import torch 
import random
class DeepThompson:
    def __init__(self, numActions, observation_dim):
        if torch.cuda.device_count() > 1:
            self.device = torch.device('cuda:1')
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.numActions = numActions
        self.epsilon = 0.1
        self.nn_model = torch.nn.Sequential(
            torch.nn.Linear(observation_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
        )
        self.heads = [torch.nn.Linear(64, 1) for _ in range(numActions)]
        for head in self.heads:
            head.to(self.device)
        self.nn_model.to(self.device)
        self.optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=0.001)
        self.loss_fn = torch.nn.MSELoss()
    def __call__(self, reward, choice, observation):
        self.optimizer.zero_grad()
        observation = torch.tensor(observation).to(self.device)
        reward = torch.tensor(reward).to(self.device)
        hidden_64 = self.nn_model(observation)
        output = self.heads[choice](hidden_64)
        loss = self.loss_fn(output, reward)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    def draw(self, observation):
        if random.random() < self.epsilon:
            return random.randint(0, self.numActions-1)
        observation = torch.tensor(observation).to(self.device)
        with torch.no_grad():
            hidden64 = self.nn_model(observation)
            output = torch.cat([head(hidden64) for head in self.heads])
            return output.argmax().item()
    def reset(self):
        pass
from tqdm import tqdm
def deepthompson_test():
    numActions = 2
    numRounds = 10000
    bandit = DeepThompson(numActions, 2)
    for i in (pbar := tqdm(range(numRounds), position=0, leave=True, dynamic_ncols=True)):
        rint = float(random.randint(0,1))
        choice = bandit.draw([rint/1.0, rint/1.0])
        if rint == 0 and choice == 0:
            reward = 1.0
        else:
            reward = 0.0
        loss = bandit(reward, choice, [rint/1.0, rint/1.0])
        pbar.set_description(f"Choice: {choice} / Loss: {loss:.2f}")
    print(bandit.draw([0.01,0.01]))
