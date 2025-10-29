import streamlit as st
import ir_datasets
import pickle
import xgboost as xgb
from utils.indexer import InvertedIndex
from utils.features import FeatureEngineer
import os
import numpy as np
import nltk
from sklearn.metrics import ndcg_score
from evaluate_standard import evaluate_all
import pandas as pd
import matplotlib.pyplot as plt

nltk.download('stopwords')

MODEL_DIR = "data/models"

st.set_page_config(page_title="Cranfield Boolean+LTR Search", layout="wide")

@st.cache_resource
def load_artifacts():
    ds = ir_datasets.load("cranfield")
    docs = {int(d.doc_id): (d.title or "") + " " + (d.text or "") for d in ds.docs_iter()}
    idx = InvertedIndex()
    idx.load("data/inverted_index.json")
    with open(MODEL_DIR + "/feature_engineer.pkl","rb") as f:
        fe = pickle.load(f)
    bst = xgb.Booster()
    bst.load_model(MODEL_DIR + "/xgb_rank_ndcg.model")
    return ds, docs, idx, fe, bst

ds, docs, idx, fe, bst = load_artifacts()

st.title("Cranfield — Boolean Shortlist + XGBoost LTR Ranking")

tab1, tab2, tab3 = st.tabs(["Search", "Performance Metrics", "About the Project"])


# -----------------------------
# ✅ Corrected metric computation
# -----------------------------
def compute_metrics(retrieved, relevant, k):
    """Compute Precision@k, Recall@k, and nDCG@k"""
    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)

    # Binary relevance list for top-k
    hits = [1 if d in relevant_set else 0 for d in retrieved_k]

    # Precision@k
    precision = np.mean(hits) if hits else 0.0

    # Recall@k
    recall = sum(hits) / len(relevant_set) if relevant_set else 0.0

    # nDCG@k
    if len(hits) > 0:
        # True relevance scores (ideal list of top-k all relevant)
        true_relevance = [1] * min(len(relevant_set), k) + [0] * (k - min(len(relevant_set), k))
        ndcg = ndcg_score([true_relevance], [hits])
    else:
        ndcg = 0.0

    return round(precision, 4), round(recall, 4), round(ndcg, 4)


# -----------------------------
# Search tab
# -----------------------------
with tab1:
    q = st.text_input("Enter query:", value="boundary layer airfoil")
    k = st.slider("Top K results", 1, 50, 10)

    if st.button("Search"):
        # Get relevant docs from qrels
        query_obj = next((x for x in ds.queries_iter() if q.lower() in x.text.lower()), None)
        if query_obj:
            qid = int(query_obj.query_id)
            relevant_docs = [int(qr.doc_id) for qr in ds.qrels_iter() if int(qr.query_id) == qid and qr.relevance > 0]
        else:
            relevant_docs = []

        if not relevant_docs:
            st.warning("⚠️ No relevance judgments found for this query in Cranfield dataset.")
            relevant_docs = []

        # --- Boolean shortlist ---
        try:
            shortlist = sorted(list(idx.eval_boolean(q)))
        except Exception as e:
            st.error(f"Boolean parse error: {e}")
            shortlist = []

        # --- LTR ranking ---
        candidate_docs = list(docs.keys())
        X = fe.make_training_data_for_query(q, candidate_docs)
        dmat = xgb.DMatrix(np.array(X))
        preds = bst.predict(dmat)
        ranked = [d for _, d in sorted(zip(preds, candidate_docs), key=lambda x: x[0], reverse=True)]

        # --- Layout ---
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Boolean Shortlist")
            st.write(f"Size: {len(shortlist)}")
            for doc_id in shortlist[:k]:
                st.markdown(f"**Doc {doc_id}**")
                st.write(docs[doc_id][:300])

        with col2:
            st.subheader("LTR Model Ranking")
            for doc_id in ranked[:k]:
                st.markdown(f"**Doc {doc_id}** (score: {preds[list(candidate_docs).index(doc_id)]:.4f})")
                st.write(docs[doc_id][:300])

        # --- Compute and show metrics comparison ---
        p_b, r_b, n_b = compute_metrics(shortlist, relevant_docs, k)
        p_l, r_l, n_l = compute_metrics(ranked, relevant_docs, k)

        st.markdown("---")
        st.subheader(f"Performance Metrics (Top {k})")

        st.table({
            "Metric": ["Precision@k", "Recall@k", "nDCG@k"],
            "Boolean": [p_b, r_b, n_b],
            "LTR Model": [p_l, r_l, n_l]
        })


# -----------------------------
# Info tab
# -----------------------------
with tab2:
    st.title("Evaluation Metrics")
    k = st.slider("Select cutoff k", 5, 30, 10)
    if st.button("Run Full Evaluation"):
        with st.spinner("Evaluating all 225 Cranfield queries..."):
            results = evaluate_all(k=k)

        df = pd.DataFrame([
            {"Metric": m, "Boolean": v[0], "LTR": v[1]} for m, v in results.items()
        ])
        st.dataframe(df.style.format(subset=["Boolean", "LTR"], formatter="{:.4f}"))

        fig, ax = plt.subplots()
        df.set_index("Metric")[["Boolean", "LTR"]].plot(kind="bar", ax=ax)
        plt.title(f"Performance Comparison (k={k})")
        plt.ylabel("Score")
        plt.xticks(rotation=0)
        st.pyplot(fig)
