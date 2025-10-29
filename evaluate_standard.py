# evaluate_standard.py
import ir_datasets
import pickle
import numpy as np
import xgboost as xgb
from utils.indexer import InvertedIndex
from utils.features import FeatureEngineer
from utils.eval_metrics import precision_at_k, mapk, ndcg_at_k
from tqdm import tqdm
import os

MODEL_DIR = "data/models"

def evaluate_all(k=10):
    ds = ir_datasets.load("cranfield")
    docs = {int(d.doc_id): (d.title or "") + " " + (d.text or "") for d in ds.docs_iter()}

    # Load index & model
    idx = InvertedIndex()
    idx.load("data/inverted_index.json")
    with open(os.path.join(MODEL_DIR, "feature_engineer.pkl"), "rb") as f:
        fe = pickle.load(f)
    bst = xgb.Booster()
    bst.load_model(os.path.join(MODEL_DIR, "xgb_rank_ndcg.model"))

    # Relevance judgments
    qrels = list(ds.qrels_iter())
    rel_dict = {}
    for qr in qrels:
        rel_dict.setdefault(int(qr.query_id), {})[int(qr.doc_id)] = int(qr.relevance)

    bool_prec, bool_map, bool_ndcg = [], [], []
    ltr_prec, ltr_map, ltr_ndcg = [], [], []

    queries = list(ds.queries_iter())
    for q in tqdm(queries, desc="Evaluating Cranfield Queries"):
        qid, qtext = int(q.query_id), q.text
        relevant = [d for d, rel in rel_dict.get(qid, {}).items() if rel > 0]

        # Boolean retrieval
        try:
            bool_docs = list(idx.eval_boolean(qtext))
        except Exception:
            bool_docs = []
        bool_prec.append(precision_at_k(bool_docs, relevant, k))
        bool_map.append(mapk([relevant], [bool_docs], k))
        bool_ndcg.append(ndcg_at_k(bool_docs, relevant, k))

        # LTR retrieval
        candidate_docs = list(docs.keys())
        X = fe.make_training_data_for_query(qtext, candidate_docs)
        dmat = xgb.DMatrix(np.array(X))
        preds = bst.predict(dmat)
        ranked_docs = [d for _, d in sorted(zip(preds, candidate_docs), key=lambda x: x[0], reverse=True)]
        ltr_prec.append(precision_at_k(ranked_docs, relevant, k))
        ltr_map.append(mapk([relevant], [ranked_docs], k))
        ltr_ndcg.append(ndcg_at_k(ranked_docs, relevant, k))

    results = {
        "Precision@{}".format(k): (np.mean(bool_prec), np.mean(ltr_prec)),
        "MAP": (np.mean(bool_map), np.mean(ltr_map)),
        "nDCG@{}".format(k): (np.mean(bool_ndcg), np.mean(ltr_ndcg))
    }

    return results
