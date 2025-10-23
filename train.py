# train.py
import ir_datasets
import pickle
import numpy as np
import xgboost as xgb
from tqdm import tqdm
from utils.indexer import InvertedIndex
from utils.features import FeatureEngineer
import os
from collections import defaultdict
from utils.eval_metrics import precision_at_k, mapk, ndcg_at_k

MODEL_DIR = "data/models"
os.makedirs(MODEL_DIR, exist_ok=True)

def build_corpus_and_index():
    ds = ir_datasets.load("cranfield")
    docs = {}
    index = InvertedIndex()
    for d in ds.docs_iter():
        doc_id = int(d.doc_id)
        text = f"{d.title or ''} {d.text or ''}"
        docs[doc_id] = text
        index.add_document(doc_id, text)
    return ds, docs, index

def prepare_training_matrix(ds, docs, fe):
    # For each query, create candidate doc list (we will consider all docs for Cranfield since small)
    X_all = []
    y_all = []
    groups = []
    qid_order = []
    for q in tqdm(list(ds.queries_iter())):
        qid = int(q.query_id)
        qtext = q.text
        # candidate docs are all docs in small collection. (can also use boolean shortlist)
        candidate_docs = list(docs.keys())
        X_q = fe.make_training_data_for_query(qtext, candidate_docs)
        # build relevance vector using qrels
        rel_map = {int(qr.doc_id): int(qr.relevance) for qr in ds.qrels_iter() if int(qr.query_id) == qid}
        y_q = [rel_map.get(d, 0) for d in candidate_docs]
        X_all.extend(X_q)
        y_all.extend(y_q)
        groups.append(len(candidate_docs))
        qid_order.append((qid, candidate_docs))
    return np.array(X_all), np.array(y_all), groups, qid_order

def train():
    ds, docs, index = build_corpus_and_index()
    print(f"Loaded {len(docs)} docs and queries: {len(list(ds.queries_iter()))}")
    # Save index
    index.save("data/inverted_index.json")
    fe = FeatureEngineer(docs)
    X, y_train, groups, qid_order = prepare_training_matrix(ds, docs, fe)
    y_train = (y_train > 0).astype(int)
    dtrain = xgb.DMatrix(X, label=y_train)
    dtrain.set_group(groups)
    params = {
        "objective": "rank:ndcg",
        "eta": 0.1,
        "max_depth": 6,
        "eval_metric": "ndcg@10",
        "verbosity": 1,
    }
    bst = xgb.train(params, dtrain, num_boost_round=150)
    bst.save_model(os.path.join(MODEL_DIR, "xgb_rank_ndcg.model"))
    # Save feature engineer and qid_order for use in app/eval
    with open(os.path.join(MODEL_DIR, "feature_engineer.pkl"), "wb") as f:
        pickle.dump(fe, f)
    with open(os.path.join(MODEL_DIR, "qid_order.pkl"), "wb") as f:
        pickle.dump(qid_order, f)
    print("Model and artifacts saved.")
    return docs, index, fe, bst, qid_order

if __name__ == "__main__":
    train()
