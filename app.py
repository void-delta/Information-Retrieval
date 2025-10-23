# app.py
import streamlit as st
import ir_datasets
import pickle
import xgboost as xgb
from utils.indexer import InvertedIndex
from utils.features import FeatureEngineer
import os
import numpy as np

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

tab1, tab2 = st.tabs(["Search", "Dataset / Model info"])

with tab1:
    q = st.text_input("Enter query:", value="boundary layer airfoil")
    mode = st.radio("Search mode:", ["LTR (model rank)", "Boolean shortlist"])
    k = st.slider("Top K results", 1, 50, 10)

    if st.button("Search"):
        if mode == "Boolean shortlist":
            try:
                shortlist = sorted(list(idx.eval_boolean(q)))
            except Exception as e:
                st.error(f"Boolean parse error: {e}")
                shortlist = []
            st.write(f"Boolean shortlist size: {len(shortlist)}")
            for doc_id in shortlist[:k]:
                st.markdown(f"**Doc {doc_id}**")
                st.write(docs[doc_id][:500])
        else:
            # LTR ranking across all docs (or you could shortlist first, then rank)
            candidate_docs = list(docs.keys())
            X = fe.make_training_data_for_query(q, candidate_docs)
            dmat = xgb.DMatrix(np.array(X)) if 'xgboost' in globals() else xgb.DMatrix(np.array(X))
            preds = bst.predict(dmat)
            ranked = [d for _, d in sorted(zip(preds, candidate_docs), key=lambda x: x[0], reverse=True)]
            st.write(f"Top {k} LTR results:")
            for doc_id in ranked[:k]:
                st.markdown(f"**Doc {doc_id}** (score: {preds[list(candidate_docs).index(doc_id)]:.4f})")
                st.write(docs[doc_id][:500])

with tab2:
    st.subheader("Dataset & Model")
    st.write("Documents:", len(docs))
    st.write("Queries:", len(list(ds.queries_iter())))
    st.write("Index loaded with N =", idx.N)
    if st.button("Show sample query/qrel"):
        q = next(ds.queries_iter())
        st.write("Query id:", q.query_id)
        st.write(q.text)
        qrels = [qr for qr in ds.qrels_iter() if int(qr.query_id) == int(q.query_id)]
        st.write("Relevance judgments (sample):", qrels[:10])
