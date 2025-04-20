# ORCAS Search Engine - Bigram Version + Topic Modeling
# Tokenization, Inverted Index, TF-IDF + Weighted Word2Vec Hybrid Search, Relevance Feedback, Topics

import os
import re
import nltk
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim import corpora, models

# Download NLTK resources
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")

# ---------------------------- Global Initialization ----------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ---------------------------- Helper Functions ----------------------------

def preprocess(text):
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"https?://\S+", "", text)
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalpha() and t not in stop_words]
    return tokens

def build_ngrams(tokens, n=2):
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def build_inverted_index(docs_tokens):
    inverted_index = defaultdict(set)
    for doc_id, tokens in docs_tokens.items():
        for token in tokens:
            inverted_index[token].add(doc_id)
    for token in inverted_index:
        inverted_index[token] = sorted(list(inverted_index[token]))
    return inverted_index

def save_unigrams(docs_tokens, output_file="unigrams.txt"):
    term_freq = defaultdict(int)
    doc_freq = defaultdict(int)

    for tokens in docs_tokens.values():
        counter = Counter(tokens)
        for token, count in counter.items():
            term_freq[token] += count
            doc_freq[token] += 1

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("word document_frequency term_frequency\n")
        for word in sorted(term_freq.keys()):
            f.write(f"{word} {doc_freq[word]} {term_freq[word]}\n")

def save_inverted_index(inverted_index, output_file="inverted_index.txt"):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("word document_id\n")
        for word in sorted(inverted_index.keys()):
            doc_list = ",".join(inverted_index[word])
            f.write(f"{word}: {doc_list}\n")

def build_tfidf(docs_texts, ngram_range=(2,2)):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=ngram_range)
    tfidf_matrix = vectorizer.fit_transform(docs_texts)
    return vectorizer, tfidf_matrix

def train_word2vec(docs_tokens):
    from gensim.models import Word2Vec
    corpus = list(docs_tokens.values())
    model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1, workers=4, seed=42)
    return model

def compute_weighted_word2vec(docs_tokens, tfidf_vectorizer, tfidf_matrix, w2v_model):
    feature_names = tfidf_vectorizer.get_feature_names_out()
    idf_scores = dict(zip(feature_names, tfidf_vectorizer.idf_))

    doc_embeddings = {}
    for idx, (doc_id, tokens) in enumerate(docs_tokens.items()):
        vectors = []
        weights = []
        for token in tokens:
            if token in w2v_model.wv and token in idf_scores:
                vectors.append(w2v_model.wv[token])
                weights.append(idf_scores[token])

        if vectors:
            weighted_vectors = np.array(vectors) * np.array(weights)[:, np.newaxis]
            embedding = np.sum(weighted_vectors, axis=0) / (np.sum(weights) + 1e-9)
            doc_embeddings[doc_id] = embedding
        else:
            doc_embeddings[doc_id] = np.zeros(w2v_model.vector_size)

    return doc_embeddings

def build_query_embedding(query, w2v_model, tfidf_vectorizer):
    tokens = build_ngrams(preprocess(query), n=2)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    idf_scores = dict(zip(feature_names, tfidf_vectorizer.idf_))

    vectors = []
    weights = []
    for token in tokens:
        if token in w2v_model.wv and token in idf_scores:
            vectors.append(w2v_model.wv[token])
            weights.append(idf_scores[token])

    if vectors:
        weighted_vectors = np.array(vectors) * np.array(weights)[:, np.newaxis]
        embedding = np.sum(weighted_vectors, axis=0) / (np.sum(weights) + 1e-9)
        return embedding
    else:
        return np.zeros(w2v_model.vector_size)

def relevance_feedback(original_query, top_docs, docs_texts, tfidf_vectorizer, top_k_terms=5):
    feedback_texts = [docs_texts[doc_id] for doc_id in top_docs if doc_id in docs_texts]
    if not feedback_texts:
        return original_query

    feedback_matrix = tfidf_vectorizer.transform(feedback_texts)
    summed_feedback = np.asarray(feedback_matrix.sum(axis=0)).flatten()

    feature_names = tfidf_vectorizer.get_feature_names_out()
    top_indices = np.argsort(summed_feedback)[-top_k_terms:]

    feedback_terms = [feature_names[idx] for idx in top_indices if feature_names[idx] not in original_query.lower().split()]
    new_query = original_query + " " + " ".join(feedback_terms)
    return new_query.strip()

def hybrid_search(query, docs, docs_tokens, docs_texts, tfidf_vectorizer, tfidf_matrix, w2v_model, doc_embeddings, alpha=0.7):
    query_vec = tfidf_vectorizer.transform([' '.join(build_ngrams(preprocess(query), n=2))])
    tfidf_scores = cosine_similarity(query_vec, tfidf_matrix)[0]
    tfidf_doc_scores = dict(zip(docs.keys(), tfidf_scores))

    query_embedding = build_query_embedding(query, w2v_model, tfidf_vectorizer)
    doc_ids = list(doc_embeddings.keys())
    doc_vecs = np.array([doc_embeddings[doc_id] for doc_id in doc_ids])
    if np.any(query_embedding):
        semantic_scores = cosine_similarity([query_embedding], doc_vecs)[0]
    else:
        semantic_scores = np.zeros(len(doc_ids))

    final_scores = {}
    for idx, doc_id in enumerate(doc_ids):
        tfidf_score = tfidf_doc_scores.get(doc_id, 0.0)
        sem_score = semantic_scores[idx]
        final_scores[doc_id] = alpha * tfidf_score + (1 - alpha) * sem_score

    ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:10]

def show_results(results, docs):
    for doc_id, score in results:
        snippet = " ".join(docs[doc_id].split()[:20])
        print(f"{doc_id}: {score:.4f}")

# ---------------------------- Main ----------------------------

def main():
    input_file = 'orcas_with_english_content_new.tsv'
    df = pd.read_csv(input_file, sep='\t', header=None, names=['DID', 'URL', 'Content'], dtype=str)
    docs = dict(zip(df['DID'], df['Content']))

    docs_tokens = {doc_id: build_ngrams(preprocess(content), n=2) for doc_id, content in docs.items()}

    # inverted_index = build_inverted_index(docs_tokens)
    # save_unigrams(docs_tokens, output_file="unigrams.txt")
    # save_inverted_index(inverted_index, output_file="inverted_index.txt")



    tfidf_vectorizer, tfidf_matrix = build_tfidf([' '.join(docs_tokens[doc_id]) for doc_id in docs], ngram_range=(2,2))
    w2v_model = train_word2vec(docs_tokens)
    doc_embeddings = compute_weighted_word2vec(docs_tokens, tfidf_vectorizer, tfidf_matrix, w2v_model)

    query = "University of Michigan"
    initial_results = hybrid_search(query, docs, docs_tokens, docs, tfidf_vectorizer, tfidf_matrix, w2v_model, doc_embeddings)
    feedback_query = relevance_feedback(query, [doc for doc, _ in initial_results], docs, tfidf_vectorizer)
    refined_results = hybrid_search(feedback_query, docs, docs_tokens, docs, tfidf_vectorizer, tfidf_matrix, w2v_model, doc_embeddings)

    print("\nInitial Results:")
    show_results(initial_results, docs)

    print("\nRefined Results with Relevance Feedback + Word2Vec:")
    show_results(refined_results, docs)

if __name__ == '__main__':
    main()
