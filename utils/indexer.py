# utils/indexer.py
import re
import json
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ps = PorterStemmer()
STOP = set(stopwords.words("english"))

token_re = re.compile(r"\b[a-zA-Z]+\b")

def tokenize(text):
    tokens = token_re.findall(text.lower())
    tokens = [ps.stem(t) for t in tokens if t not in STOP]
    return tokens

class InvertedIndex:
    def __init__(self):
        # term -> {doc_id: [positions]}
        self.postings = defaultdict(lambda: defaultdict(list))
        self.doc_lengths = {}
        self.N = 0

    def add_document(self, doc_id, text):
        tokens = tokenize(text)
        self.N += 1
        self.doc_lengths[doc_id] = len(tokens)
        for pos, t in enumerate(tokens):
            self.postings[t][doc_id].append(pos)

    def get_postings(self, term):
        return self.postings.get(term, {})

    def save(self, path):
        # careful: convert inner mappings to lists for JSON
        serial = {
            "postings": {t: {str(d): v for d, v in docs.items()} for t, docs in self.postings.items()},
            "doc_lengths": self.doc_lengths,
            "N": self.N
        }
        with open(path, "w") as f:
            json.dump(serial, f)

    def load(self, path):
        with open(path) as f:
            serial = json.load(f)
        self.postings = defaultdict(lambda: defaultdict(list))
        for t, docs in serial["postings"].items():
            for d, pl in docs.items():
                self.postings[t][int(d)] = pl
        self.doc_lengths = {int(k): v for k, v in serial["doc_lengths"].items()}
        self.N = serial["N"]

    # Boolean query evaluation (supports simple AND, OR, NOT and parentheses)
    # We'll implement a recursive descent parser for expressions composed of terms, AND, OR, NOT.
    def _term_set(self, term):
        term = ps.stem(term.lower())
        if term in STOP:
            return set()
        return set(self.postings.get(term, {}).keys())

    def eval_boolean(self, expr):
        """
        expr: string, e.g. "aerodynamics AND NOT propeller OR (wing AND lift)"
        returns set of doc_ids
        """
        tokens = self._tokenize_expr(expr)
        self._tokens = tokens
        self._pos = 0
        res = self._parse_or()
        return res

    def _tokenize_expr(self, expr):
        # simple tokenizer returning terms and operators
        parts = re.findall(r'\(|\)|AND|OR|NOT|\b[a-zA-Z]+\b', expr)
        return [p for p in parts if p.strip()]

    def _peek(self):
        if self._pos < len(self._tokens): return self._tokens[self._pos]
        return None

    def _eat(self, tok=None):
        cur = self._peek()
        if tok and cur != tok:
            raise ValueError(f"Expected {tok}, got {cur}")
        self._pos += 1
        return cur

    def _parse_or(self):
        left = self._parse_and()
        while True:
            if self._peek() == "OR":
                self._eat("OR")
                right = self._parse_and()
                left = left.union(right)
            else:
                break
        return left

    def _parse_and(self):
        left = self._parse_not()
        while True:
            if self._peek() == "AND":
                self._eat("AND")
                right = self._parse_not()
                left = left.intersection(right)
            else:
                break
        return left

    def _parse_not(self):
        if self._peek() == "NOT":
            self._eat("NOT")
            sub = self._parse_atom()
            # all docs minus sub
            all_docs = set(self.doc_lengths.keys())
            return all_docs - sub
        else:
            return self._parse_atom()

    def _parse_atom(self):
        tok = self._peek()
        if tok == "(":
            self._eat("(")
            res = self._parse_or()
            self._eat(")")
            return res
        else:
            term = self._eat()
            return self._term_set(term)
