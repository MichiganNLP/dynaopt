import numpy as np 
def get_entropy(scores):
    # normalize the scores
    scores = scores / np.sum(scores)
    # calculate the entropy
    return -np.sum(scores * np.log2(scores))
def get_avg_mean(scores):
    return np.mean(scores)
def harmonic_score(scores):
    """
    Calculates the harmonic score of a given data set.
    """
    avg_mean = get_avg_mean(scores)
    entropy = get_entropy(scores)
    return 2 * avg_mean * entropy / (avg_mean + entropy)
def main():
    scores = np.array([0.1, 0.2, 0.9])
    print(harmonic_score(scores))   
    scores = np.array([0.4, 0.4, 0.4])
    print(harmonic_score(scores))  
    scores = np.array([-0.1, 0.2, -0.4])
    print(harmonic_score(scores))  
if __name__ == "__main__":
    main()