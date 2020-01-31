# Scripts
## Python scripts
These scripts facilitate using NLP and graphical analysis methods with MAG data.
All of them require Pandas, the graph-related ones also require NetworkX 2.0+

| Script                          | Description                                                                                                                                                              |
| --------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `abstracts.py`                  | Converts inverted indexes to a (optionally gzipped) JSON, CSV, TSV or pickle file containing the papers' abstracts.                                                      |
| `citation_graph.py`             | Creates a GEXF or pickled NetworkX graph file for the paper citation network with papers, authors, journals or affiliations as nodes and citations as edges.             |
| `coauthorship_graph.py`         | Creates a GEXF or pickled NetworkX graph file for the coauthorship network with authors or affiliations as nodes and coauthorships as edges. This may take over an hour. |
| `field_of_study_graph.py`       | Creates a GEXF or pickled NetworkX graph file for the MAG ChildFieldOfStudy field of study hierarchy.                                                                    |

## U-SQL scripts
These are the scripts used to generate the datasets on an Azure Data Lake Service instance.
