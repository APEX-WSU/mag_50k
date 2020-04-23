#!/usr/bin/env python3
'''
Produce CSV files easily readable by Postgres' /copy
'''
import gzip
import json
import tensorflow_hub as hub
import pandas as pd
import sys
import progressbar
from nltk.tokenize import sent_tokenize
import tensorflow as tf


def load_abstracts(filename):
    with open(filename) as file:
        data = json.load(file)
        paper_ids = [key for key in data.keys()]
        abstracts = [data[key] for key in paper_ids]
        paper_ids = [int(paper_id) for paper_id in paper_ids]
        del data
    return paper_ids, abstracts


def embed_abstracts(abstracts, paper_ids, prefix):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    print('Loading USE')
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def embed_abstract(abstract, i, paper_id, progress):
        sentences = sent_tokenize(abstract)
        sentence_embeddings = embed(sentences).numpy()
        progress.update(i)
        return str(paper_id) + '\t' + '{' + ','.join([str(v) for v in sentence_embeddings.mean(axis=0)]) + '}\n'

    def write_embeddings(embeddings, filename):
        print(f'writing to {filename}')
        with open(filename, 'w') as file:
            file.writelines(embeddings)

    print('Computing embeddings')
    embeddings = []
    suffix = 0
    with progressbar.ProgressBar(max_value=len(abstracts)) as bar:
        for i, (paper_id, abstract) in enumerate(zip(paper_ids, abstracts)):
            if len(embeddings) >= 250000:
                write_embeddings(embeddings, f'{prefix}_{suffix}.tsv')
                suffix += 1
                embeddings = []
            embeddings.append(embed_abstract(abstract, i, paper_id, bar))
    suffix += 1
    write_embeddings(embeddings, f'{prefix}_{suffix}.tsv')


def main():
    infilename = sys.argv[1]
    outfilename = sys.argv[2]
    print(f'Loading abstracts from {infilename}')
    paper_ids, abstracts = load_abstracts(infilename)
    embed_abstracts(abstracts, paper_ids, outfilename)


if __name__ == '__main__':
    main()
