# evaluate.py
import pickle
import xgboost as xgb
import numpy as np
from utils.eval_metrics import precision_at_k, mapk, ndcg_at_k
from utils.indexer import InvertedIndex
from collections import defaultdict

MODEL_DIR = "data/models"

def load_artifacts():
    with open(MODEL_DIR + "/feature_engineer.pkl","rb") as f:
        fe = pickle.load(f)
    with open(MODEL_DIR + "/qid_order.pkl","rb") as f:
        qid_order = pickle.load(f)
    bst = xgb.Booster()
    bst.load_model(MODEL_DIR + "/xgb_rank_ndcg.model")
    idx = InvertedIndex()
    idx.load("data/inverted_index.json")
    return fe, qid_order, bst, idx

def evaluate(ds, docs):
    fe, qid_order, bst, idx = load_artifacts()
    X_all_pred = []
    all_actuals = []
    all_preds = []
    precisions = []
    ndcgs = []
    # For each query
    for q in ds.queries_iter():
        qid = int(q.query_id)
        qtext = q.text
        # candidate docs = all docs (same order as during training)
        candidate_docs = [pair[1] for pair in qid_order if pair[0]==qid]
        if candidate_docs:
            candidate_docs = candidate_docs[0]
        else:
            candidate_docs = list(docs.keys())
        X = fe.make_training_data_for_query(qtext, candidate_docs)
        dmat = xgb.DMatrix(np.array(X))
        preds = bst.predict(dmat)
        ranked = [d for _, d in sorted(zip(preds, candidate_docs), key=lambda x: x[0], reverse=True)]
        # actual relevant set from qrels
        rels = {int(qr.doc_id) for qr in ds.qrels_iter() if int(qr.query_id) == qid and int(qr.relevance) > 0}
        all_actuals.append(rels)
        all_preds.append(ranked)
        precisions.append(precision_at_k(rels, ranked, 10))
        ndcgs.append(ndcg_at_k(rels, ranked, 10))
    avg_prec = sum(precisions)/len(precisions)
    avg_ndcg = sum(ndcgs)/len(ndcgs)
    avg_map = mapk(all_actuals, all_preds, k=10)
    print("Precision@10:", avg_prec)
    print("nDCG@10:", avg_ndcg)
    print("MAP@10:", avg_map)
    return avg_prec, avg_ndcg, avg_map

if __name__ == "__main__":
    import ir_datasets
    ds = ir_datasets.load("cranfield")
    docs = {int(d.doc_id): (d.title or "") + " " + (d.text or "") for d in ds.docs_iter()}
    evaluate(ds, docs)
