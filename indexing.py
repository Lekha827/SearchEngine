from collections import defaultdict, Counter

def build_inverted_index(docs_tokens):
    inverted_index = defaultdict(set)
    for doc_id, tokens in docs_tokens.items():
        for token in tokens:
            inverted_index[token].add(doc_id)
    return {token: sorted(list(doc_ids)) for token, doc_ids in inverted_index.items()}

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
