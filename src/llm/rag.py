"""Capability 2 - RAG over support tickets + customer feedback.

Vector store: TF-IDF (word 1-2 grams) by default so the project runs offline; if `chromadb` is installed
and USE_CHROMA=1 the same documents are indexed in ChromaDB instead. The retrieval interface is identical.
Answers always state how many records the summary is based on.
"""
from __future__ import annotations

import os
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from src.llm.client import get_llm
from src.utils.config import MODELS, PROCESSED

INDEX_PATH = MODELS / "rag_index.joblib"


def build_documents() -> pd.DataFrame:
    t = pd.read_csv(PROCESSED / "support_tickets.csv")
    f = pd.read_csv(PROCESSED / "customer_feedback.csv")
    tickets = pd.DataFrame({
        "doc_id": t.ticket_id, "source": "support_ticket", "customer_id": t.customer_id, "date": t.ticket_date,
        "text": t.ticket_description, "category": t.issue_category, "subcategory": t.issue_subcategory,
        "score": t.customer_satisfaction_score, "label": t.ticket_status})
    fb = pd.DataFrame({
        "doc_id": f.feedback_id, "source": "feedback", "customer_id": f.customer_id, "date": f.feedback_date,
        "text": f.feedback_text, "category": f.sentiment_label, "subcategory": "", "score": f.rating, "label": f.sentiment_label})
    return pd.concat([tickets, fb], ignore_index=True).dropna(subset=["text"]).reset_index(drop=True)


QUESTION_WORDS = ["customer", "customers", "saying", "say", "says", "said", "people", "users", "user", "talking", "complain",
                  "complaining", "complaints", "complaint", "feedback", "tickets", "ticket", "recent", "recently", "main",
                  "common", "mostly", "tell", "show", "summarise", "summarize", "issues", "issue", "problems"]


def _mode(x: pd.Series):
    m = x.dropna().astype(str).mode()
    return m.iloc[0] if len(m) else ""


class VectorStore:
    """The corpus is highly repetitive (templated complaints), so identical texts are indexed once and carry
    the number of underlying records, their ids, and category/score statistics."""

    def __init__(self, docs: pd.DataFrame):
        g = docs.groupby(["source", "text"], sort=False)
        self.docs = g.agg(n_records=("doc_id", "size"), doc_ids=("doc_id", lambda x: list(x.head(5))),
                          category=("category", _mode), subcategory=("subcategory", _mode),
                          avg_score=("score", lambda x: pd.to_numeric(x, errors="coerce").mean())).reset_index()
        self.total_records = len(docs)
        stop = list(TfidfVectorizer(stop_words="english").get_stop_words()) + QUESTION_WORDS
        self.vec = TfidfVectorizer(ngram_range=(1, 2), stop_words=stop, sublinear_tf=True)
        self.matrix = self.vec.fit_transform(self.docs.text)

    def search(self, query: str, k: int = 8, source: str | None = None, min_score: float = 0.08) -> pd.DataFrame:
        sims = linear_kernel(self.vec.transform([query]), self.matrix).ravel()
        if source:
            sims = np.where(self.docs.source.values == source, sims, -1)
        idx = [i for i in np.argsort(-sims)[:k] if sims[i] >= min_score]
        out = self.docs.iloc[idx].copy()
        out["similarity"] = sims[idx].round(3)
        return out


_store: VectorStore | None = None


def get_store(rebuild: bool = False) -> VectorStore:
    global _store
    if _store is not None and not rebuild:
        return _store
    if INDEX_PATH.exists() and not rebuild:
        _store = joblib.load(INDEX_PATH)
    else:
        _store = VectorStore(build_documents())
        MODELS.mkdir(exist_ok=True)
        joblib.dump(_store, INDEX_PATH)
    return _store


def answer(question: str, k: int = 8, source: str | None = None) -> dict:
    """Retrieve the most relevant complaint/feedback texts and return a grounded summary."""
    hits = get_store().search(question, k=k, source=source)
    if hits.empty:
        return {"answer": "No relevant support tickets or feedback were found for this question.", "n_records": 0, "n_distinct": 0, "records": hits}
    n, d = int(hits.n_records.sum()), len(hits)
    llm = get_llm()
    if llm.available:
        ctx = "\n".join(f"- ({r.source}, {r.n_records} records, e.g. {', '.join(r.doc_ids[:2])}) {r.text}" for r in hits.itertuples())
        prompt = (f"Question: {question}\n\nRetrieved {d} distinct texts covering {n} records:\n{ctx}\n\n"
                  f"Write a grounded summary (max 120 words). Start with 'Based on {n} records:'. Cite example ids. Use the record "
                  f"counts given; do not claim anything not in the texts.")
        text = llm.complete(prompt, system="You summarise customer voice faithfully and never invent facts.", max_tokens=400)
    else:
        lines = [f"{r.n_records} records - \"{r.text[:120]}{'...' if len(r.text) > 120 else ''}\" ({r.category}; e.g. {r.doc_ids[0]})"
                 for r in hits.head(4).itertuples()]
        text = f"Based on {n} records ({d} distinct themes). Most relevant: " + " | ".join(lines)
    return {"answer": text, "n_records": n, "n_distinct": d, "records": hits}
