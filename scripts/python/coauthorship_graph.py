#!/usr/bin/env python3
import sys
import os
import itertools
import pandas as pd
import networkx as nx
from joblib import Parallel, delayed


def edges_from_paper(paper_id, df, node_key, other_key):
    data = df.loc[[paper_id]].drop_duplicates(node_key).set_index(node_key).to_dict('index')
    edges = []
    for a, b in itertools.combinations(data, 2):
        edges.append((a, b, paper_id, {
                          'PaperId': paper_id,
                          'AuthorSequenceNumber': data[a]['AuthorSequenceNumber'],
                          other_key: data[a][other_key]
                      }))
        edges.append((b, a, paper_id, {
                          'PaperId': paper_id,
                          'AuthorSequenceNumber': data[b]['AuthorSequenceNumber'],
                          other_key: data[b][other_key]
                      }))
    return edges


def get_coauthorship_graph(input_dir, by='author', n_jobs=None, verbose=0):
    node_key = 'AuthorId' if by == 'author' else 'AffiliationId'
    other_key = 'AffiliationId' if by == 'author' else 'AuthorId'
    g = nx.MultiDiGraph()
    paa = pd.read_csv(os.path.join(input_dir, 'paper_author_affiliations.tsv'), dtype={'AffiliationId': 'Int64'}, sep='\t', index_col=0)
    paa = paa[paa.index.duplicated()]  # only papers with more than one author
    
    if by == 'affiliation':
        paa = paa.dropna()
    else:
        paa = paa.fillna(value={'AffiliationId': -1})
    
    node_filename = 'authors.tsv' if by == 'author' else 'affiliations.tsv'
    nodes = pd.read_csv(os.path.join(input_dir, node_filename), sep='\t', index_col=0, dtype={'LastKnownAffiliationId': 'Int64'})  # for node labels
    
    if by == 'author':
        nodes.fillna(value={'LastKnownAffiliationId': -1})

    if verbose:
        print('Generating edgelist:')
    edges = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(edges_from_paper)(i, paa, node_key, other_key) for i in pd.unique(paa.index))
    
    if verbose:
        print('Loading edgelist:')
    for edges_ in edges:
        g.add_edges_from(edges_)
    
    if verbose:
        print('Setting node attributes')
    
    nx.set_node_attributes(g, nodes.to_dict('index')) 
    return g


def main():
    input_dir = sys.argv[1]
    output_filename = sys.argv[2]
    by = sys.argv[3]
    graph = get_coauthorship_graph(input_dir, by, -1, 1)
    print('Writing file')
    _, extension = os.path.splitext(output_filename)
    if extension == '.gexf':
        nx.write_gexf(graph, output_filename)
    else:  # elif extension == '.pkl': only .pkl and .gexf are supported
        nx.write_gpickle(graph, output_filename)

if __name__ == '__main__':
    main()

