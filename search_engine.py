# ORCAS Search Engine - Full Pipeline
# Implements: Tokenization, Inverted Index, TF-IDF, Unigram, N-gram, Word2Vec, PageRank/HITS, and Relevance Feedback

import os
import re
import nltk
import math
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec

# Download required NLTK resources
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")

# ---------------------------- Initialization ----------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ---------------------------- Document Preprocessing ----------------------------
def preprocess(text):
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"https?://\S+", "", text)
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalpha() and t not in stop_words]
    return tokens

# ---------------------------- Inverted Index + Unigram ----------------------------
def build_inverted_index(docs):
    index = defaultdict(list)
    doc_lengths = {}
    for doc_id, text in docs.items():
        tokens = preprocess(text)
        doc_lengths[doc_id] = len(tokens)
        tf = Counter(tokens)
        for term, freq in tf.items():
            index[term].append((doc_id, freq))
    return index, doc_lengths

# ---------------------------- TF-IDF Model ----------------------------
def build_tfidf_matrix(docs):
    vectorizer = TfidfVectorizer(stop_words='english')
    doc_texts = [docs[doc_id] for doc_id in docs]
    tfidf_matrix = vectorizer.fit_transform(doc_texts)
    return vectorizer, tfidf_matrix

# ---------------------------- N-Gram Language Model ----------------------------
def build_ngram_model(docs, n=2):
    ngram_freq = defaultdict(Counter)
    for doc_id, text in docs.items():
        tokens = preprocess(text)
        ngrams = zip(*[tokens[i:] for i in range(n)])
        for ng in ngrams:
            prefix = " ".join(ng[:-1])
            ngram_freq[prefix][ng[-1]] += 1
    return ngram_freq

# ---------------------------- Word2Vec Semantic Similarity ----------------------------
def train_word2vec_model(docs):
    corpus = [preprocess(text) for text in docs.values()]
    model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1, workers=4)
    return model

def get_semantic_sim(query, docs, model):
    query_tokens = preprocess(query)
    query_vecs = [model.wv[w] for w in query_tokens if w in model.wv]
    if not query_vecs:
        return []
    query_vec = np.mean(query_vecs, axis=0)
    similarities = {}
    for doc_id, text in docs.items():
        tokens = preprocess(text)
        doc_vecs = [model.wv[t] for t in tokens if t in model.wv]
        if doc_vecs:
            doc_vec = np.mean(doc_vecs, axis=0)
            sim = cosine_similarity([query_vec], [doc_vec])[0][0]
            similarities[doc_id] = sim
    return sorted(similarities.items(), key=lambda x: x[1], reverse=True)

# ---------------------------- PageRank / HITS ----------------------------
def compute_pagerank(graph, damping=0.85, max_iter=100):
    N = len(graph)
    pr = {node: 1/N for node in graph}
    for _ in range(max_iter):
        new_pr = {}
        for node in graph:
            new_pr[node] = (1 - damping) / N + damping * sum(pr[nei] / len(graph[nei]) for nei in graph[node])
        pr = new_pr
    return pr

# ---------------------------- Relevance Feedback ----------------------------
def relevance_feedback(original_query, top_docs, docs):
    feedback_tokens = []
    for doc_id in top_docs:
        feedback_tokens.extend(preprocess(docs[doc_id]))
    augmented_query = original_query + " " + " ".join(feedback_tokens[:10])
    return augmented_query

# ---------------------------- Search Pipeline ----------------------------
def search(query, docs, tfidf_vectorizer, tfidf_matrix, w2v_model=None, use_semantics=False):
    query_vec = tfidf_vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix)[0]
    doc_scores = dict(zip(docs.keys(), scores))
    ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

    if use_semantics and w2v_model:
        sem_sim = get_semantic_sim(query, docs, w2v_model)
        return sem_sim[:10]

    return ranked[:10]

# ---------------------------- Main Usage with ORCAS Subset ----------------------------
def main():
    input_file = 'orcas_with_english_content_subset.tsv'
    df = pd.read_csv(input_file, sep='\t', header=None, names=['DID', 'URL', 'Content'], dtype=str)
    docs = dict(zip(df['DID'], df['Content']))

    # Build all models
    index, doc_lengths = build_inverted_index(docs)
    tfidf_vectorizer, tfidf_matrix = build_tfidf_matrix(docs)
    w2v_model = train_word2vec_model(docs)
    ngram_model = build_ngram_model(docs)

    # Example search
    query = "Michigan"
    initial_results = search(query, docs, tfidf_vectorizer, tfidf_matrix)
    feedback_query = relevance_feedback(query, [doc for doc, _ in initial_results], docs)
    refined_results = search(feedback_query, docs, tfidf_vectorizer, tfidf_matrix, w2v_model, use_semantics=True)

    print("Initial Results:")
    for doc_id, score in initial_results:
        print(f"{doc_id}: {score:.4f}")

    print("\nRefined Results with Relevance Feedback + Word2Vec:")
    for doc_id, score in refined_results:
        print(f"{doc_id}: {score:.4f}")

if __name__ == '__main__':
    main()
