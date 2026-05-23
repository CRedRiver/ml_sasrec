import numpy as np

def hit_rate_at_k(recommended, ground_truth, k=10):
    recommended = recommended[:k]
    return int(len(set(recommended) & set(ground_truth)) > 0)

def dcg_at_k(recommended, ground_truth, k=10):
    recommended = recommended[:k]
    dcg = 0
    for i, item in enumerate(recommended):
        if item in ground_truth:
            dcg += 1 / np.log2(i + 2)
    return dcg

def ndcg_at_k(recommended, ground_truth, k=10):
    ideal_dcg = dcg_at_k(ground_truth, ground_truth, k)
    if ideal_dcg == 0:
        return 0
    return dcg_at_k(recommended, ground_truth, k) / ideal_dcg

def mrr_at_k(recommended, ground_truth, k=10):
    recommended = recommended[:k]
    for i, item in enumerate(recommended):
        if item in ground_truth:
            return 1.0 / (i + 1)
    return 0.0