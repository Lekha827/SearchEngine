import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from vectorization import build_query_embedding

def hybrid_search(query, docs, docs_tokens, docs_texts, tfidf_vectorizer, tfidf_matrix, w2v_model, doc_embeddings, preprocess, alpha=0.7, only_relevant=True):
    query_vec = tfidf_vectorizer.transform([query])
    tfidf_scores = cosine_similarity(query_vec, tfidf_matrix)[0]
    tfidf_doc_scores = dict(zip(docs.keys(), tfidf_scores))

    query_embedding = build_query_embedding(query, preprocess, w2v_model, tfidf_vectorizer)
    doc_ids = list(doc_embeddings.keys())
    doc_vecs = np.array([doc_embeddings[doc_id] for doc_id in doc_ids])
    semantic_scores = cosine_similarity([query_embedding], doc_vecs)[0] if np.any(query_embedding) else np.zeros(len(doc_ids))

    final_scores = {doc_id: alpha * tfidf_doc_scores.get(doc_id, 0.0) + (1 - alpha) * semantic_scores[idx] 
                    for idx, doc_id in enumerate(doc_ids)}

    query_words = set(query.lower().split())
    if only_relevant:
        filtered_scores = {
            doc_id: score for doc_id, score in final_scores.items()
            if any(word in docs_texts[doc_id].lower() for word in query_words)
        }
    else:
        filtered_scores = final_scores 

    return sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)[:10]

def relevance_feedback(original_query, top_docs, docs_texts, tfidf_vectorizer, top_k_terms=5):
    feedback_texts = [docs_texts[doc_id] for doc_id in top_docs if doc_id in docs_texts]
    if not feedback_texts:
        return original_query
    feedback_matrix = tfidf_vectorizer.transform(feedback_texts)
    summed_feedback = np.asarray(feedback_matrix.sum(axis=0)).flatten()
    feature_names = tfidf_vectorizer.get_feature_names_out()
    top_indices = np.argsort(summed_feedback)[-top_k_terms:]
    feedback_terms = [feature_names[i] for i in top_indices if feature_names[i] not in original_query.lower().split()]
    return original_query + " " + " ".join(feedback_terms)

def format_results(results, docs_df, max_words=30):
    """
    Format search results as tuples (doc_id, score, url, snippet)
    """
    formatted = []
    for doc_id, score in results:
        row = docs_df.loc[docs_df['DID'] == doc_id].iloc[0]
        snippet = " ".join(row['Content'].split()[:max_words])
        formatted.append((doc_id, f"{score:.4f}", row['URL'], snippet))
    return formatted
