import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec

def build_tfidf(docs_texts):
    vectorizer = TfidfVectorizer(stop_words='english')
    matrix = vectorizer.fit_transform(docs_texts)
    return vectorizer, matrix

def train_word2vec(docs_tokens):
    corpus = list(docs_tokens.values())
    return Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1, workers=4, seed=42)

def compute_weighted_word2vec(docs_tokens, tfidf_vectorizer, tfidf_matrix, w2v_model):
    feature_names = tfidf_vectorizer.get_feature_names_out()
    idf_scores = dict(zip(feature_names, tfidf_vectorizer.idf_))
    doc_embeddings = {}
    for idx, (doc_id, tokens) in enumerate(docs_tokens.items()):
        vectors, weights = [], []
        for token in tokens:
            if token in w2v_model.wv and token in idf_scores:
                vectors.append(w2v_model.wv[token])
                weights.append(idf_scores[token])
        if vectors:
            weighted = np.array(vectors) * np.array(weights)[:, None]
            embedding = np.sum(weighted, axis=0) / (np.sum(weights) + 1e-9)
        else:
            embedding = np.zeros(w2v_model.vector_size)
        doc_embeddings[doc_id] = embedding
    return doc_embeddings

def build_query_embedding(query, preprocess, w2v_model, tfidf_vectorizer):
    tokens = preprocess(query)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    idf_scores = dict(zip(feature_names, tfidf_vectorizer.idf_))
    vectors, weights = [], []
    for token in tokens:
        if token in w2v_model.wv and token in idf_scores:
            vectors.append(w2v_model.wv[token])
            weights.append(idf_scores[token])
    if vectors:
        weighted = np.array(vectors) * np.array(weights)[:, None]
        return np.sum(weighted, axis=0) / (np.sum(weights) + 1e-9)
    return np.zeros(w2v_model.vector_size)
