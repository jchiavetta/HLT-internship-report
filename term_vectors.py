import pickle
import gensim
from gensim import *
import sklearn
from sklearn import *
import numpy as np


### This script was used to load and run similarity queries using term-based vectors created with Gensim. The following data
### in /wiki-sim-search-master/data/ was made with command-line arguments that took the full Wikipedia dump as input
### (Multistream version required). I was helped in this process by the following blog post and associated Github repo:
### http://mccormickml.com/2017/02/22/concept-search-on-wikipedia/


# Contains .index, a numpy sparse matrix of 300-feature LSI article vectors. takes a few minutes to load and about 4.5 gb of RAM.
# Meant for iterating a query over the entire matrix, but we'll pull out individual articles instead.
index = similarities.MatrixSimilarity.load('./full_wiki/wiki-sim-search-master/data/lsi_index.mm')
# Retrieve article vectors by .index[id].
id_to_titles = utils.unpickle('./full_wiki/wiki-sim-search-master/data/bow.mm.metadata.cpickle')
titles_to_id = utils.unpickle('./full_wiki/wiki-sim-search-master/data/titles_to_id.pickle')

titles_to_id_lower = {}
id_to_titles_lower = {}
for i in titles_to_id:
	titles_to_id_lower[i.lower()] = titles_to_id[i]
	id_to_titles_lower[titles_to_id[i]] = i.lower()

# id -> word dict used to convert context to bag of words vectors.
id2word = gensim.corpora.Dictionary.load_from_text('./full_wiki/wiki-sim-search-master/data/dictionary.txt')
# lsi word weights used to convert bag of words to lsi feature vectors.
lsi = gensim.models.LsiModel.load('./full_wiki/wiki-sim-search-master/data/lsi.lsi_model')

# These are files I created in lum_files.py
with open('new_id_senses.txt', 'rb') as a, open('node_context_with_count.txt', 'rb') as b, open('anchor_matched_nodes.txt', 'rb') as c:
	# id -> list of article names that have linked to that node id.
	id_senses = pickle.load(a)
	# id -> concatenated string of all the evidence text for nodes neighboring the id node.
	node_context = pickle.load(b)
	# id -> string, only the node strings that perfect match anchor text.
	nodes = pickle.load(c)

# reverse of node dict, string -> id
r_nodes = {}
for i in nodes:
	r_nodes[nodes[i]] = i


def similarity(node):
	""" takes a node id, assuming it has at least one potential sense, and outputs a list of similarities.
	These indicate overlap in meaning between a sense and the context evidence of the node.
	"""

	# This is the pseudo-document for the given node. It is converted to LSI to match the Wikipedia article vectors.
	doc = node_context[node][0]
	doc_bow = id2word.doc2bow(doc.lower().split())
	doc_lsi = lsi[doc_bow]
	doc_lsi = [i[1] for i in doc_lsi]
	doc_array = np.asarray(doc_lsi)

	# Possible senses for the given node, articles that linked to the anchor text represented by the node.
	queries = []
	for sense in id_senses[node]:
		if sense in titles_to_id_lower:
			# This isn't the node id, rather that of the corresponding article in the lsi matrix		
			query_id = titles_to_id_lower[sense]
			query_lsi = index.index[query_id, :]
			# (vector, sense name)
			queries.append((query_lsi, sense))

	# After creating/retrieving the relevant LSI vectors, compare each one and return a sorted list of the best results.
	results = []
	for query in queries:
		sim = sklearn.metrics.pairwise.cosine_similarity(doc_array.reshape(1, -1), query[0].reshape(1, -1))
		results.append((sim, query[1], node_context[node][1]))

	results = sorted(results, reverse=True)
	return results


