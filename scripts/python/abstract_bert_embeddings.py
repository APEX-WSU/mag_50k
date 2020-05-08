#!/usr/bin/env python3
'''
Get sentence embeddings using a BERT-style transformer model
'''
import gzip
import json
import pandas as pd
import numpy as np
import sys
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from sentence_transformers import models, SentenceTransformer


def load_abstracts(filename):
    with open(filename) as file:
        data = json.load(file)
        paper_ids = [key for key in data.keys()]
        abstracts = [data[key] for key in paper_ids]
        paper_ids = [int(paper_id) for paper_id in paper_ids]
        del data
    return paper_ids, abstracts


def embed_abstracts(abstracts, model_name, term_pooling, sentence_pooling):
    print('Loading models')
    word_embedding_model = models.Transformer(model_name)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=(term_pooling == 'mean'),
                                   pooling_mode_cls_token=(term_pooling == 'cls'),
                                   pooling_mode_max_tokens=(term_pooling == 'max'))
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    def embed_abstract(abstract):
        sentences = sent_tokenize(abstract)
        sentence_embeddings = model.encode(sentences)
        if sentence_pooling == 'mean':
            sentence_embeddings = np.array(sentence_embeddings).mean(axis=0)
        elif sentence_pooling == 'max':
            sentence_embeddings = np.array(sentence_embeddings).max(axis=0)
        else:
            sentence_embeddings = np.array(sentence_embeddings).sum(axis=0)
        return sentence_embeddings
    
    print('Computing embeddings')
    embeddings = [embed_abstract(abstract) for abstract in tqdm(abstracts)]
    return embeddings


def main():
    infilename = sys.argv[1]
    outfilename = sys.argv[2]
    model_name = sys.argv[3]
    term_pooling = sys.argv[4]  # 'mean', 'max', 'cls'
    sentence_pooling = sys.argv[5]  # 'mean', 'max', 'sum'

    print(f'Loading abstracts from {infilename}')
    paper_ids, abstracts = load_abstracts(infilename)
    embeddings = embed_abstracts(abstracts, model_name, term_pooling, sentence_pooling)    
    print(f'Exporting to {outfilename}')
    if outfilename.endswith('h5'):
        pd.DataFrame(embeddings, index=paper_ids).to_hdf(outfilename, 'abstract_embeddings')
    elif outfilename.endswith('csv'):
        pd.DataFrame(embeddings, index=paper_ids).to_csv(outfilename)
    else:
        pd.DataFrame(embeddings, index=paper_ids).to_pickle(outfilename)


if __name__ == '__main__':
    main()
