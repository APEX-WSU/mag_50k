#!/usr/bin/env python3
import sys
import os
import pandas as pd
import networkx as nx


def get_paper_citation_graph(input_dir):
    paper_references = pd.read_csv(os.path.join(input_dir, 'paper_references.tsv'), sep='\t')
    dtypes = {
        'FirstPage': 'str',
        'LastPage': 'str',
        'JournalId': 'Int64',
        'Volume': 'Int64',
        'Issue': 'Int64',
        'ConferenceSeriesId': 'Int64',
        'ConferenceInstanceId': 'Int64',
        'FamilyId': 'Int64'
    }
    papers = pd.read_csv(os.path.join(input_dir, 'papers.tsv'), sep='\t', index_col=0, dtype=dtypes)
    g = nx.DiGraph()
    g.add_nodes_from([(key, value) for key, value in papers.to_dict('index').items()])
    g.add_edges_from([tuple(row) for row in paper_references.values])
    return g
    

def get_author_citations(input_dir, by):
    paper_author_affiliations = pd.read_csv(os.path.join(input_dir, 'paper_author_affiliations.tsv'),
                                           sep='\t',
                                           dtype={'AffiliationId': 'Int64'})
    paper_author_affiliations = paper_author_affiliations.fillna(value={'AffiliationId': -1})
    paper_author_affiliations = paper_author_affiliations[['PaperId', 'AuthorId', 'AuthorSequenceNumber', 'AffiliationId']]
    paper_references = pd.read_csv(os.path.join(input_dir, 'paper_references.tsv'),
                                  sep='\t')
    authors = pd.read_csv(os.path.join(input_dir, 'authors.tsv'),
                          sep='\t',
                          index_col=0,
                          dtype={'LastKnownAffiliationId': 'Int64'})
    authors = authors.fillna(value={'LastKnownAffiliationId': -1})
    affiliations = pd.read_csv(os.path.join(input_dir, 'affiliations.tsv'), sep='\t', index_col=0)
    author_citations = paper_references.merge(paper_author_affiliations, on='PaperId', how='inner')
    author_citations = author_citations.merge(paper_author_affiliations,
                                              left_on='PaperReferenceId',
                                              right_on='PaperId',
                                              suffixes=('', '_'),
                                              how='inner').drop('PaperId_', axis=1)
    author_citations = author_citations.rename({
        key: f'Reference{key[:-1]}'
        for key in ('AffiliationId_', 'AuthorSequenceNumber_', 'AuthorId_')
    }, axis=1)
    
    return author_citations, authors if by=='author' else affiliations
 


def get_edge_tuple(row, by='author'):
    if by == 'affiliation':
        node_keys = ('AffiliationId', 'ReferenceAffiliationId')
    else:
        node_keys = ('AuthorId', 'ReferenceAuthorId')
    metadata = {col: row[col] for col in row.index if col not in node_keys}
    return row[node_keys[0]], row[node_keys[1]], metadata


def get_citation_graph(input_dir, by='author'):
    if by == 'paper':
        return get_paper_citation_graph(input_dir)
    author_citations, node_labels = get_author_citations(input_dir, by)
    g = nx.DiGraph()
    g.add_nodes_from([(key, value) for key, value in node_labels.to_dict('index').items()])
    edges = [get_edge_tuple(row, by) for _, row in author_citations.iterrows()]
    g.add_edges_from(edges)
    return g


def main():
    input_dir = sys.argv[1]
    output_filename = sys.argv[2]
    by = sys.argv[3]
    graph = get_citation_graph(input_dir, by)
    print('Writing file')
    _, extension = os.path.splitext(output_filename)
    if extension == '.gexf':
        nx.write_gexf(graph, output_filename)
    else:  # elif extension == '.pkl': only .pkl and .gexf are supported
        nx.write_gpickle(graph, output_filename)


if __name__ == '__main__':
    main()

