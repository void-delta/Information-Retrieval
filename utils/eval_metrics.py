# utils/eval_metrics.py
import math

def precision_at_k(relevant_set, ranked_list, k):
    topk = ranked_list[:k]
    if not topk:
        return 0.0
    return sum(1 for d in topk if d in relevant_set) / k

def apk(actual, predicted, k):
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    if not actual:
        return 0.0
    return score / min(len(actual), k)

def mapk(list_of_actuals, list_of_predicted, k=10):
    return sum(apk(a, p, k) for a, p in zip(list_of_actuals, list_of_predicted)) / len(list_of_actuals)

def dcg_at_k(relevant_list_binary, k):
    dcg = 0.0
    for i in range(k):
        rel = relevant_list_binary[i] if i < len(relevant_list_binary) else 0
        dcg += (2 ** rel - 1) / math.log2(i + 2)
    return dcg

def ndcg_at_k(relevant_set, ranked_list, k):
    # build binary relevance list
    rels = [1 if d in relevant_set else 0 for d in ranked_list[:k]]
    dcg = dcg_at_k(rels, k)
    # ideal DCG
    ideal = sorted(rels, reverse=True)
    idcg = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0
