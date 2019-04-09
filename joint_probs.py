import pickle


### This is the script I used to get the commonness / probability scores discussed in the paper.


# I created these files in lum_files.py.
with open('./full_wiki/joint_counts_lite.txt', 'rb') as a, open('./matched_nodes.txt', 'rb') as b:
	# dictionary of (sense, anchor) pair -> count of how often that combination appeared throughout Wikipedia.
	joints = pickle.load(a)
	# lum.ai nodes that perfect match with anchor text so that we can retrieve the list of possible senses.
	nodes = pickle.load(b)

	# This one node in particular was NaN, I do not know why.
	del nodes[559908309]

	nodes = {i: nodes[i].lower() for i in nodes}

# We want to look up each anchor and get the list of potential senses along with that pair's frequency count.
# Format is anchor -> [(sense, count), ...].
# Pair is a key (sense, anchor) with a count value
anchors = {}
for pair in joints:
	if pair[1] not in anchors:
		anchors[pair[1]] = [(pair[0], joints[pair])]
	else:
		anchors[pair[1]].append((pair[0], joints[pair]))

# reverse of node dict for string lookup, string -> id
r_nodes = {}
for i in nodes:
	r_nodes[nodes[i]] = i


def probability(node):
	""" takes a node ID, assuming it's in anchors, and outputs a sorted list 
	of the most probable senses based on link counts.
	"""
	senses = anchors[nodes[node]]
	total = 0
	# pair is a tuple (sense, count)
	for pair in senses:
		# how many times was ANY article linked to from this anchor
		total += pair[1]
	results = []
	for pair in senses:
		# how many times was THIS article linked to from this anchor
		local = pair[1]
		# Given this anchor, what's the chance that this article was linked to.
		prob = local / total * 100
		results.append((pair[0], round(prob, 2)))
	results = sorted(results, key=lambda x: x[1], reverse=True)
	return results
