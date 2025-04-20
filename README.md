ORCAS Search Engine (Optimized Version)
=======================================

A hybrid search engine built on the ORCAS dataset that combines classical Information Retrieval techniques with semantic similarity from Word2Vec embeddings. It supports:

- Tokenization & preprocessing
- Inverted index generation
- TF-IDF vectorization
- Weighted Word2Vec embedding for documents
- Hybrid search (TF-IDF + Word2Vec)
- Query expansion using relevance feedback
- Persistent storage of unigrams and inverted index with metadata

Project Structure
-----------------
orcas_with_english_content_new.tsv  # Input dataset
search_engine.py                    # Main Python script
unigrams.txt                        # Output: unigrams with term & doc frequencies
inverted_index.txt                  # Output: inverted index file


Features
--------
Preprocessing
- HTML and URL removal
- Tokenization (via NLTK)
- Stopword removal and lemmatization

Indexing
- Inverted Index (word → docID list)
- Unigram frequency stats (document frequency and total term frequency)

Search Model
- TF-IDF Vectorization: Captures importance of terms
- Word2Vec Embeddings: Captures semantic similarity using weighted term vectors
- Hybrid Scoring: Combines TF-IDF and Word2Vec scores using a tunable alpha parameter

Relevance Feedback
- Top documents from initial search used to expand query with top-K TF-IDF terms

Setup & Run
-----------
1. Install Dependencies

    pip install pandas nltk scikit-learn gensim

2. Download NLTK Resources (auto-downloaded, or manually):

    import nltk
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")

3. Input File Format

TSV file with 3 columns:
DocumentID   URL   EnglishContent

Update the filename in script if needed with complete path:
input_file = 'orcas_with_english_content_new.tsv'

How It Works
------------
1. Preprocessing: Clean and tokenize documents.
2. Indexing: Create inverted index and unigram stats.
3. Modeling:
   - Compute TF-IDF matrix
   - Train Word2Vec on tokenized documents
   - Compute weighted document embeddings
4. Search:
   - User inputs query
   - Results ranked using a hybrid of TF-IDF and semantic similarity
   - Query is expanded via feedback and re-ranked

Example Output
--------------
Initial Results:
D60280: 0.6540
D340987: 0.4393
...

Refined Results with Relevance Feedback + Word2Vec:
D60280: 0.7092
D562024: 0.6097
...

Output Files
------------
unigrams.txt:
Format:
word document_frequency term_frequency
emotion 8 21
retrieval 15 31
...

inverted_index.txt:
Format:
word: doc1,doc2,doc3
search: doc2,doc7
...


References
----------
- ORCAS Dataset: https://microsoft.github.io/orcas/
- TF-IDF - scikit-learn: https://scikit-learn.org/stable/modules/feature_extraction.html
- Word2Vec - Gensim: https://radimrehurek.com/gensim/models/word2vec.html