#!/usr/bin/env python3

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


def embed_abstracts(abstracts):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
   
    print('Loading USE')
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    
    def embed_abstract(abstract, i, progress):
        sentences = sent_tokenize(abstract)
        sentence_embeddings = embed(sentences).numpy()
        progress.update(i)
        return sentence_embeddings.mean(axis=0)
    
    print('Computing embeddings')
    with progressbar.ProgressBar(max_value=len(abstracts)) as bar:
        embeddings = [embed_abstract(abstract, i, bar) for i, abstract in enumerate(abstracts)]
    return embeddings


def main():
    infilename = sys.argv[1]
    outfilename = sys.argv[2]
    print(f'Loading abstracts from {infilename}')
    paper_ids, abstracts = load_abstracts(infilename)
    embeddings = embed_abstracts(abstracts)    
    print(f'Exporting to {outfilename}')
    pd.DataFrame(embeddings, index=paper_ids).to_pickle(outfilename)


if __name__ == '__main__':
    main()
