import os
from flask import Flask, render_template, request
import pandas as pd

from preprocessing import preprocess
from indexing import build_inverted_index
from vectorization import build_tfidf, train_word2vec, compute_weighted_word2vec
from search import hybrid_search, relevance_feedback, format_results

app = Flask(__name__)

# ---- Global variables shared across routes ----
docs = {}
docs_tokens = {}
tfidf_vectorizer = None
tfidf_matrix = None
w2v_model = None
doc_embeddings = {}
df = None  # The full dataframe with URL + Content

# ---- Load everything only once, not on Flask reloader ----
if __name__ == '__main__':
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":  # Ensures it doesn't run twice
        print("Loading and preprocessing documents...")
        df = pd.read_csv("orcas_with_english_content_new.tsv", sep='\t', header=None, names=['DID', 'URL', 'Content'], dtype=str)
        docs.update(dict(zip(df['DID'], df['Content'])))
        docs_tokens.update({doc_id: preprocess(content) for doc_id, content in docs.items()})
        tfidf_vectorizer, tfidf_matrix = build_tfidf([docs[doc_id] for doc_id in docs])
        w2v_model = train_word2vec(docs_tokens)
        doc_embeddings.update(compute_weighted_word2vec(docs_tokens, tfidf_vectorizer, tfidf_matrix, w2v_model))

@app.route('/', methods=['GET', 'POST'])
def index():
    global docs, docs_tokens, tfidf_vectorizer, tfidf_matrix, w2v_model, doc_embeddings, df

    results_initial = []
    results_refined = []
    query = ""
    only_relevant = True  # default behavior

    if request.method == 'POST':
        query = request.form['query']
        only_relevant = request.form.get('only_relevant') == 'on'

        initial = hybrid_search(query, docs, docs_tokens, docs, tfidf_vectorizer, tfidf_matrix, w2v_model, doc_embeddings, preprocess, only_relevant)
        feedback_q = relevance_feedback(query, [doc for doc, _ in initial], docs, tfidf_vectorizer)
        refined = hybrid_search(feedback_q, docs, docs_tokens, docs, tfidf_vectorizer, tfidf_matrix, w2v_model, doc_embeddings, preprocess, only_relevant)

        results_initial = format_results(initial, df)
        results_refined = format_results(refined, df)

    return render_template("index.html",
                       query=query,
                       results_initial=results_initial if not only_relevant else [],
                       results_refined=results_refined,
                       only_relevant=only_relevant)



if __name__ == '__main__':
    app.run(debug=True)
