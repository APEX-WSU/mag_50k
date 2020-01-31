#!/usr/bin/env python3
import sys
import os
import pandas as pd
import networkx as nx


def get_field_of_study_graph(input_dir):
    field_of_study_children = pd.read_csv(os.path.join(input_dir, 'field_of_study_children.tsv'), sep='\t')
    fields_of_study = pd.read_csv(os.path.join(input_dir, 'fields_of_study.tsv'), sep='\t', index_col=0)
    g = nx.DiGraph()
    g.add_nodes_from([(key, value) for key, value in fields_of_study.to_dict('index').items()])
    g.add_edges_from([tuple(row) for row in field_of_study_children.values])
    return g
    

def main():
    input_dir = sys.argv[1]
    output_filename = sys.argv[2]
    graph = get_field_of_study_graph(input_dir)
    print('Writing file')
    _, extension = os.path.splitext(output_filename)
    if extension == '.gexf':
        nx.write_gexf(graph, output_filename)
    else:  # elif extension == '.pkl': only .pkl and .gexf are supported
        nx.write_gpickle(graph, output_filename)


if __name__ == '__main__':
    main()

