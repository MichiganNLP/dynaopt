import scipy
from scipy import stats
import numpy as np
import pandas as pd
def moving_average(x, w):
    """
    Moving average of a sequence
    :param x: array_like, sequence of values
    :param w: int, window size
    :return: array_like, moving average of x
    """
    return np.convolve(x, np.ones(w), 'valid') / w
def exponential_moving_average(x, alpha):
    """
    Exponential moving average of a sequence
    :param x: array_like, sequence of values
    :param alpha: float, smoothing factor
    :return: array_like, exponential moving average of x
    """
    x = pd.Series(x)
    smoothed =  x.ewm(alpha=alpha).mean()
    return np.array(smoothed).tolist()
def student_t_test(x, y):
    """
    Student's t-test for two independent samples of scores
    :param x: array_like, sample of scores
    :param y: array_like, sample of scores
    :return: t-value and p-value
    """
    return stats.ttest_ind(x, y)
def paired_student_t_test(x, y):
    """
    Student's t-test for two paired samples of scores
    :param x: array_like, sample of scores
    :param y: array_like, sample of scores
    :return: t-value and p-value
    """
    return stats.ttest_rel(x, y)
def wilcoxon_signed_rank_test(x, y):
    """
    Wilcoxon signed-rank test for two paired samples of scores
    :param x: array_like, sample of scores
    :param y: array_like, sample of scores
    :return: t-value and p-value
    """
    return stats.wilcoxon(x, y)
def main():
    x = "0.7376685844	0.6846897442	0.7123568159	0.7445031563	0.7098522955".split()
    x = [float(i) for i in x]
    y = "0.8537637012	0.6963595123	0.7467978729	0.7289618752	0.821170706".split()
    y = [float(i) for i in y]
    print(paired_student_t_test(x, y))
if __name__ == '__main__':
    main()