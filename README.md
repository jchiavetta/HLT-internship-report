# HLT_internship_report
Code used in the research report for my internship with lum.ai. My task was to work on a large knowledge graph and use natural language processing skills to disambiguate the graph entities and link them to unique Wikipedia articles. For more details on the task and my approach/results, please see the included report.

Consists of four python scripts:
* lum_files.py - processes the data from the company
* joint_probs.py - baseline commonness measure
* term_vectors.py - similarity measure based on vectors created with the Gensim package
* link_vectors.py. - similarity measure based on vectors created with just Wikipedia article links
