#!/usr/bin/env python3
import sys
import os
import itertools
import pandas as pd
import networkx as nx
from joblib import Parallel, delayed

def edges_from_paper(paper_id, df):
    data = df.loc[[paper_id]].drop_duplicates('AuthorId').set_index('AuthorId').to_dict('index')
    edges = []
    for author1, author2 in itertools.combinations(data, 2):
        edges.append((author1, author2, {
                          'PaperId': paper_id,
                          'AuthorSequenceNumber': data[author1]['AuthorSequenceNumber'],
                          'AffiliationId': data[author1]['AffiliationId']
                      }))
        edges.append((author2, author1, {
                          'PaperId': paper_id,
                          'AuthorSequenceNumber': data[author2]['AuthorSequenceNumber'],
                          'AffiliationId': data[author2]['AffiliationId']
                      }))
    return edges


def get_coauthorship_graph(input_dir, n_jobs=None, verbose=0):
    g = nx.MultiDiGraph()
    paa = pd.read_csv(os.path.join(input_dir, 'paper_author_affiliations.tsv'), dtype={'AffiliationId': 'Int64'}, sep='\t', index_col=0)
    paa = paa[paa.index.duplicated()]  # only papers with more than one author
    paa = paa.fillna(value={'AffiliationId': -1})
    authors = pd.read_csv(os.path.join(input_dir, 'authors.tsv'), sep='\t', index_col=0)  # for node labels
    if verbose:
        print('Generating edgelist:')
    edges = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(edges_from_paper)(i, paa) for i in pd.unique(paa.index))
    if verbose:
        print('Loading edgelist:')
    for edges_ in edges:
        g.add_edges_from(edges_)
    if verbose:
        print('Setting node attributes')
    nx.set_node_attributes(g, authors.to_dict('index')) 
    return g


#def get_affiliation_coauthorship_graph(filename):
#    paa = pd.read_csv(filename, dtype={'AffiliationId': 'Int64'}, sep='\t', index_col=0)

def main():
    input_dir = sys.argv[1]
    output_filename = sys.argv[2]
    graph = get_coauthorship_graph(input_dir, -1, 1)
    print('Writing file')
    _, extension = os.path.splitext(output_filename)
    if extension == '.gexf':
        nx.write_gexf(graph, output_filename)
    else:  # elif extension == '.pkl': only .pkl and .gexf are supported
        nx.write_gpickle(graph, output_filename)

if __name__ == '__main__':
    main()

