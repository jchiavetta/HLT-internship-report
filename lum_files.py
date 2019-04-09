import pandas as pd
import numpy as np
from tqdm import tqdm

### This script was used to process the three CSV files I was given by lum.ai, and then to gather the links
### from Wikipedia that I used to create the joint probabilities used in joint_probs.py. It also contains
### the code to create the pseudo-documents discussed in the paper and used to create LSI vectors in term_vectors.py.


# The nodes of the lum.ai graph, each with a unique ID.
df = pd.read_csv('lum-ai-intern/concepts.csv')

# creates a dictionary mapping node ids to node content strings e.g. {2976004250: 'CXCR3', 4198194237: 'zonulin'}
nodes = {}
for i in df.values:
	nodes[i[0]] = i[1]

# The edges of the lum.ai graph, each with a unique ID.
df2 = pd.read_csv('lum-ai-intern/influences.csv')

# Creates a dictionary mapping edge ids to edge-node triples e.g. {4192111153: ('increases', 2976004250, 4198194237)} aka "CXCR3 increases zonulin".
edges = {}
for i in df2.values:
	edges[i[0]] = ((i[3], i[4], i[5]))

# The contextual evidence from a medical paper from which lum.ai extracted an edge and corresponding node entities.
df3 = pd.read_csv('lum-ai-intern/evidence-new.csv')
# creates a dictionary mapping edge ids to source of node mentions e.g. {4192111153: 'Initially gut... induced CXCR3 activated upregulation of zonulin...}
# this takes a few minutes and a lot of RAM
evidence = {}
for i in df3.values:
	evidence[i[1]] = i[7]


##### creating psuedo-documents from edge context #####

# Creates a dictionary mapping matched node ids to list of edge IDs that contain the node (either direction).
# Matched nodes are all of the exact string matches from node titles to wikipedia article titles, only 55,000 of them.
for i in tqdm(edges):
	if edges[i][1] in matches:
		node_context[edges[i][1]].append(i)
	if edges[i][2] in matches:
		node_context[edges[i][2]].append(i)


# Takes the dict above and makes a new one concatenating the evidence strings for each node.
context_text = {}
for node in tqdm(context):
    temp = ''
    for edge in context[node]:
    	temp += ' ' + evidence[edge]
    context_text[node] = temp

# Alternate version of the above that includes the number of neighboring nodes as a diagnostic.
context_text = {}
for node in tqdm(context):
    temp = ''
    neighbor_count = 0
    for edge in context[node]:
    	neighbor_count += 1
    	temp += ' ' + evidence[edge]
    context_text[node] = (temp, neighbor_count)

##### Gathering links from Wikipedia to be used in joint_probs.py. #####


# Here we can just go through line by line and greedy-select every link.
i = 0
links = []
with open('enwiki-latest-pages-articles.xml', 'r') as f:
	for line in f:
		ms = re.findall('\[\[.*?\]\]', line)
		[links.append(i) for i in ms]
		i += 1
		if i % 10000000 == 0:
			print(str(i) + ' lines processed') 


# Takes the list of wiki links above and builds a dict with joint counts for each (sense, anchor) tuple.
i = 0
joint = {}
with open('links_raw.txt', 'r') as f:
	for line in f:
		line = line.strip()
		pipe = re.search('(.*)\|(.*)', line)
		# Links may or may not have a pipe, as discussed in the paper. [[Anarchism in Brazil|Brazil]] vs. 
		# [[Spanish Civil War]]; in the latter, the anchor text is an exact match with the article title.
		if pipe:
			if (pipe.group(1), pipe.group(2)) in joint:
				joint[(pipe.group(1), pipe.group(2))] += 1
			else:
				joint[(pipe.group(1), pipe.group(2))] = 1
		else:
			if (line, line) in joint:
				joint[(line, line)] += 1
			else:
				joint[(line, line)] = 1
		if len(joint) % 1000000 == 0:
			print(str(len(joint)) + ' links processed')

senses = {}
# Dict, anchor -> list of senses (articles) linked to from the anchor.
# Pair is a (sense, anchor) tuple
for pair in tqdm(joints):
	if pair[1] in senses:
		senses[pair[1]].append(pair[0])
	else:
		senses[pair[1]] = [pair[0]]

anchors = set([i[1] for i in joint])

# Keeps only the lum nodes that are a perfect string match with an anchor text; only about 55,000.
matched_nodes = {}
for i in nodes:
	if str(nodes[i]).lower() in anchors:
		matched_nodes[i] = nodes[i]




			