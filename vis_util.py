import matplotlib.pyplot as plt
import json
from bandit_alg import Exp3
import fire, glob
import ast
def draw_bandit_distribution(weight_history, idx, prefix="none"):
    weight_history = [[w / sum(weights) for w in weights] for weights in weight_history]
    plt.figure()
    colors = [ "red", "blue", "orange", "green"]
    if len(weight_history[0]) == 4:
        colors = ['red', '#1f77b4', '#ff7f0e', '#2ca02c']
    else:
        colors = [ '#1f77b4', '#ff7f0e', '#2ca02c']
    for i in range(len(weight_history[0])):
        plt.plot([w[i] for w in weight_history], color=colors[i])
    plt.xlabel("Number of Rounds")
    plt.ylabel("Probability")
    if len(weight_history[0]) == 4:
        plt.title("Bandit Arm Distribution History")
        plt.legend(["Do Nothing", "Reflection", "Fluency", "Coherence", "MI"])
        plt.savefig(f"{prefix}_bandit_arm_distribution_{idx}.png")
    else:
        plt.title("Reward Weight History")
        plt.legend(["Reflection", "Fluency", "Coherence"])
        # plt.savefig(f"dynaopt_reward_weight_history_{idx}.png")
        plt.savefig(f"{prefix}_reward_weight_history_{idx}.png")
    return
def main(path = 'voutputs/con_scst_contextual_MI_rl_2023_10_03_10_10_04/bandit_weight_history.json',idx=0, prefix="none"):
    with open(path, 'r') as f:
        weight_history = json.load(f)
    draw_bandit_distribution(weight_history, idx, prefix)
def draw_reward_history():
    with open('dorb_rewards', 'r') as f:
        # read the text file
        rewards = f.readlines()
        dorb_history = []
        for r in rewards:
            r = r.split("39m")[-1]
            r = ast.literal_eval(r)
            dorb_history.append(r)
    with open('dynaopt_rewards', 'r') as f:
        # read the text file
        rewards = f.readlines()
        dynaopt_history = []
        for r in rewards:
            r = r.split("39m")[-1]
            r = ast.literal_eval(r)
            dynaopt_history.append(r)
    labels = [ "DORB", "DynaOpt"]
    reward_names = [ "Reflection", "Fluency", "Coherence"]
    # visualize weight history with pyplot
    plt.figure()
    plt.plot(dorb_history)
    plt.xlabel("Number of Rounds")
    plt.ylabel("Reward")
    plt.title("DORB Reward History")
    plt.legend(reward_names)
    # plt.show()
    plt.savefig(f"dorb_reward_history.png")
    plt.figure()
    plt.plot(dynaopt_history)
    plt.xlabel("Number of Rounds")
    plt.ylabel("Reward")
    plt.title("DynaOpt Reward History")
    plt.legend(reward_names)
    # plt.show()
    plt.savefig(f"dynaopt_reward_history.png")
if __name__ == "__main__":
    draw_reward_history()
    files = glob.glob("voutputs/*con*/pmf_history.json")
    files = sorted(files)
    for i,f in enumerate(files):
        print(f)
        prefix="con"
        main(f, i, prefix)
    # fire.Fire(main)