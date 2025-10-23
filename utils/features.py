# utils/features.py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import math
from utils.indexer import tokenize

class FeatureEngineer:
    def __init__(self, docs):
        """
        docs: dict doc_id -> text
        """
        self.docs = docs
        self.doc_ids = list(docs.keys())
        self.texts = [docs[d] for d in self.doc_ids]

        # TF-IDF (for cosine similarity)
        self.tfidf = TfidfVectorizer(token_pattern=r"\b[a-zA-Z]+\b", stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(self.texts)
        # Map from doc_id -> row index in tfidf matrix
        self.docid_to_idx = {d: i for i, d in enumerate(self.doc_ids)}

        # Precompute term frequencies for BM25 and doc length
        self.term_freqs = {}
        self.doc_len = {}
        self.N = len(docs)
        self.df = Counter()
        for d, text in docs.items():
            toks = tokenize(text)
            self.doc_len[d] = len(toks)
            tf = Counter(toks)
            self.term_freqs[d] = tf
            for t in tf.keys():
                self.df[t] += 1

        # avg doc length
        self.avgdl = np.mean(list(self.doc_len.values())) if self.doc_len else 0

    def tfidf_cosine(self, query, doc_id):
        q_vec = self.tfidf.transform([query])
        d_idx = self.docid_to_idx[doc_id]
        d_vec = self.tfidf_matrix[d_idx]
        # cosine:
        num = q_vec.dot(d_vec.T).data
        if len(num) == 0:
            return 0.0
        denom = np.linalg.norm(q_vec.data) * np.linalg.norm(d_vec.data)
        return float(num[0] / denom) if denom != 0 else 0.0

    def bm25(self, query, doc_id, k1=1.5, b=0.75):
        tokens = tokenize(query)
        score = 0.0
        dl = self.doc_len[doc_id]
        for t in tokens:
            if t not in self.df:
                continue
            df = self.df[t]
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1e-8)
            f = self.term_freqs[doc_id].get(t, 0)
            denom = f + k1 * (1 - b + b * dl / self.avgdl)
            score += idf * ((f * (k1 + 1)) / (denom + 1e-8))
        return score

    def query_doc_overlap(self, query, doc_id):
        q_tokens = set(tokenize(query))
        d_tokens = set(self.term_freqs[doc_id].keys())
        if not q_tokens: return 0.0
        return len(q_tokens & d_tokens) / len(q_tokens)

    def make_feature_vector(self, query, doc_id):
        f1 = self.tfidf_cosine(query, doc_id)
        f2 = self.bm25(query, doc_id)
        f3 = self.query_doc_overlap(query, doc_id)
        f4 = self.doc_len[doc_id]
        return [f1, f2, f3, f4]

    def make_training_data_for_query(self, query, candidate_docs):
        """
        candidate_docs: list of doc_ids to create features for
        returns X matrix and doc_ids order
        """
        X = [self.make_feature_vector(query, d) for d in candidate_docs]
        return X
