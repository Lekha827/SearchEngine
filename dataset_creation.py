import pandas as pd
import csv
import requests
from bs4 import BeautifulSoup
from langdetect import detect

def extract_url_content(url):
    """Fetches and extracts visible English text content from a URL."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script and style elements
        for tag in soup(["script", "style"]):
            tag.decompose()

        text = soup.get_text(separator=' ', strip=True)
        text = text.replace('\n', ' ').replace('\r', ' ')

        # Detect language and ignore if not English
        if detect(text) != 'en':
            return None

        return text
    except Exception:
        return None

def process_chunk(chunk_df):
    """Processes a chunk of the DataFrame and returns rows with English content."""
    processed = []
    for _, row in chunk_df.iterrows():
        content = extract_url_content(row['URL'])
        if content:
            processed.append((row['DID'], row['URL'], content))
    return pd.DataFrame(processed, columns=['DID', 'URL', 'Content'])

def main():
    input_file = "C:/Users/bolli/Desktop/Lekha/Second sem MS/Text Mining CIS 536/orcas_subset.tsv"
    output_file = 'orcas_with_english_content.tsv'
    chunk_size = 500
    chunk_count = 0


    # Read and process file in chunks
    for chunk in pd.read_csv(input_file, sep='\t', chunksize=chunk_size, usecols=[2, 3], names=['DID', 'URL'], header=0):
        chunk_count += 1
        processed_chunk = process_chunk(chunk)
        processed_chunk.to_csv(
            output_file,
            sep='\t',
            mode='a',
            index=False,
            header=False,
            quoting=csv.QUOTE_MINIMAL,
            escapechar='\\'
        )
        print(f"Chunks processed: {chunk_count}")

if __name__ == '__main__':
    main()
