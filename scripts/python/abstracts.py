#!/usr/bin/env python3
import json
import sys
import os
import pandas as pd
import gzip


def uninvert_index(inverted_index):
    data = json.loads(inverted_index)
    tokens = ['' for _ in range(data['IndexLength'])]
    for key, value in data['InvertedIndex'].items():
        for v in value:
            tokens[v] = key
    return ' '.join(tokens)


def main():
    inverted_index_filename = sys.argv[1]
    abstracts_filename = sys.argv[2]
    df = pd.read_csv(inverted_index_filename, sep='\t', index_col=0)
    df['Abstract'] = df.IndexedAbstract.apply(uninvert_index)
    basename, extension = os.path.splitext(abstracts_filename)
    if extension == '.csv':
        df[['Abstract']].to_csv(abstracts_filename)
    elif extension == '.json':
        df['Abstract'].to_json(abstracts_filename, 'index')
    elif extension == '.gz':
        _, extension2 = os.path.splitext(basename)
        with gzip.open(abstracts_filename, 'wt') as outfile:
            if extension2 == '.csv':
                df[['Abstract']].to_csv(outfile)
            elif extension2 == '.json':
                df['Abstract'].to_json(outfile, 'index')
            else:
                df[['Abstract']].to_csv(outfile, sep='\t')
    elif extension == '.pkl':
        df[['Abstract']].to_pickle(abstracts_filename)
    else:
        df[['Abstract']].to_csv(abstracts_filename, sep='\t')

if __name__ == '__main__':
    main()
