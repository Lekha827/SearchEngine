import pandas as pd

def extract_subset(input_path, output_path, start_line=150000, num_lines=250000):
    # Skip header + first 50000 lines, then read next 50000
    df = pd.read_csv(input_path, sep='\t', skiprows=range(1, start_line + 1), nrows=num_lines)

    # Set the correct column names
    df.columns = ['QID', 'Q', 'DID', 'URL']

    # Remove duplicate document IDs
    df = df.drop_duplicates(subset='DID')

    # Save to output
    df.to_csv(output_path, sep='\t', index=False)
    print(f"Saved {len(df)} unique rows to {output_path}")

def main():
    full_file = 'C:/Users/bolli/Desktop/Lekha/Second sem MS/Text Mining CIS 536/orcas.tsv'
    subset_file = 'C:/Users/bolli/Desktop/Lekha/Second sem MS/Text Mining CIS 536/orcas_subset.tsv'
    extract_subset(full_file, subset_file)
    print(f"Extracted first 50000 lines to {subset_file}")

if __name__ == '__main__':
    main()