import xml.etree.ElementTree as ET
from bz2file import BZ2File
import re
from tqdm import tqdm
import pickle
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import math


### This script is what I used to create and run similarity queries using link-based vectors. Please compare
### with the other script, which served the same purpose but for term-based vectors.


##### The following section is how I created the files used. It only needed to be run once. #########

# The dump of the English language Simple Wikipedia
path = './simplewiki-20180501-pages-articles-multistream.xml.bz2'
with BZ2File(path) as file:
	tree = ET.parse(file)

root = tree.getroot()

for i in root.iter('{http://www.mediawiki.org/xml/export-0.10/}title'):
	print(i.text)

#{http://www.mediawiki.org/xml/export-0.10/}page

print(root.find('.//{http://www.mediawiki.org/xml/export-0.10/}text'))

#iter() automatically searches through ALL subelements
#find and findall() require xpath expressions. './/' searches all subelements

#Successfully creates a title > text dictionary, e.g.
#{'April': '{{monththisyear|4}}\n\'\'\'April\'\'\' is the 4th [[month]] of the [[year]], and comes between [[March]] and [[May]]...}
test_dict = {}
for page in root.iter('{http://www.mediawiki.org/xml/export-0.10/}page'):
	m = page.find('{http://www.mediawiki.org/xml/export-0.10/}title').text
	t = page.find('.//{http://www.mediawiki.org/xml/export-0.10/}text').text
	test_dict[m] = t

########## Above was a test with simple wiki, below is the full wiki. ####################

import xml.etree.ElementTree as ET
from bz2file import BZ2File
import re
i = 0
# This is the full Wikipedia dump. Multistream version was necessary for Gensim but here it doesn't matter.
path = './full_wiki/enwiki-20170820-pages-articles-multistream.xml.bz2'
with open('article_links.txt', 'w') as f:
	with BZ2File(path) as file:
		tree = ET.iterparse(file)
		for events, elem in tree:
			if elem.tag == "{http://www.mediawiki.org/xml/export-0.10/}page":
				# Once we've found a an article (page) we extract both the article title and its body text.
				title = elem.find('{http://www.mediawiki.org/xml/export-0.10/}title').text
				text = elem.find('.//{http://www.mediawiki.org/xml/export-0.10/}text').text
				if text and title:
					# finding all links within the article text
					ms = re.findall('\[\[(.*?)\]\]', text)
					if ms:
						# Storing the links in a comma-separated file with the article title as the
						# first entry in each new line.
						f.write(title + ',')
						[f.write(i + ',') for i in ms]
						f.write('\n')
						i += 1
						if i % 10000 == 0:
							print(str(i) + " pages processed")
			elem.clear()

# Reading the article and its associated links into a dictionary.
article_dict = {}
with open('./article_links.txt', 'r') as f:
	for line in f:
		raw = line.split(',')
		# Space constraints, and many articles with under 50 links are low-quality.
		if len(raw) >= 50:
			title = raw[0]
			article_dict[title] = raw[1:]
			if len(article_dict) % 100000 == 0:
				print(len(article_dict))


# This block takes the raw links and gets rid of user pages and redirects, in addition to
# removing links we don't want.
count = 0
stops = ['File:', 'Image:', 'Category:', 'Wikipedia:', 'User:']
with open('article_links_cleaned.txt', 'w') as w:
	with open('./article_links.txt', 'r') as f:
		for line in f:
			raw = line.split(',')
			if len(raw) >= 20:
				new_links = list(filter(lambda x: remove_stops(x, stops), raw))
				if new_links:
					[w.write(link + ',') for link in new_links]
					# For some reason articles are separated by a line with a single comma.
					w.write('\n')
					count += 1
					if count % 100000 == 0:
						print(count)


#This gets rid of the newlines with only a single comma that resulted from the above.
with open('article_links_fixed.txt', 'w') as w:
	with open('./article_links_cleaned.txt', 'r') as f:
		for line in f:
			if len(line) > 5:
				new = line[:-2] # new line plus trailing comma
				w.write(new)
				w.write('\n')

def remove_stops(link, stops):
	""" Meant to get rid of user pages and other types that we don't want."""
	for word in stops:
		if word in link:
			return False
	return True

# This dict stores article -> all the articles that linked to it.
incoming = {}
count = 0
with open('article_links_fixed.txt', 'r') as f:
	for line in f:
		raw = line.split(',')
		title = raw[0]
		for link in raw[1:]:
			pipe = re.search('(.*)\|.*', link)
			# We only want the article senses here, so this discards the anchor text.
			if pipe:
				raw[raw.index(link)] = pipe.group(1)
		for link in raw:
			if link in incoming:
				# Note the key is the link, not the title, because we're measuring incoming links.
				incoming[link].append(title)
			else:
				incoming[link] = [title]
		count += 1
		# These following hacks were necessary because I kept getting segmentation faults.
		if count % 1000000 == 0:
			print('removing duplicates')
			for entry in incoming:
				incoming[entry] = list(set(incoming[entry]))
		if count % 6000000 == 0:
			print('removing entries with length less than ten')
			for entry in incoming:
				if len(incoming[entry]) < 10:
					incoming[entry] = []

########## The following section is what I ran each time to get my similarity score results. #########

# The link > articles dictionary created above.
with open('incoming.pickle', 'rb') as f:
	incoming = pickle.load(f)

# The test nodes with artificially created links using Wikifier.
with open('wikifier_nodes.pickle', 'rb') as f:
	wiki_nodes = pickle.load(f)

# Test nodes: spider, low temperature, darkness, marrow, welfare.
# Incoming dict will already have some of these terms, so we need to keep track of the lum nodes
# with a unique name.
wiki_nodes_unique = {}
for i in wiki_nodes:
	wiki_nodes_unique[i + '_node'] = wiki_nodes[i]

# Heavily reducing size of dictionary due to space constraints / segmentation faults.
prune = []
for i in incoming:
	if len(incoming[i]) <500:
		prune.append(i)
for i in prune:
	del incoming[i]

# This is only necessary to match the test nodes, as Wikifier produces lowercase results.
incoming_lower = {}
for i in tqdm(incoming):
	temp = []
	for j in incoming[i]:
		temp.append(j.lower())
	incoming_lower[i] = temp

# Test nodes have the same format as the entries in incoming, so we add them
# directly to the dict so they can get an ID and be included in the LSA matrix.
for i in wiki_nodes_unique:
	incoming_lower[i] = wiki_nodes_unique[i]

# This sets up unique IDs from incoming to use for the matrix.
count = 0
article_to_id = {}
for i in incoming_lower:
	article_to_id[i] = count
	count += 1
id_to_article = {}
for i in article_to_id:
	id_to_article[article_to_id[i]] = i

# Creating the matrix of article counts. each cell = whether or not that column article linked to the row article, one-hot.
length = len(article_to_id)
# This can fail due to memory error if there are too many entries in incoming! It works with pruning <20 aka 1.2m entries.
inc_matrix = np.zeros((length, length))
for i in tqdm(range(length)):
	for j in incoming_lower[id_to_article[i]]:
		if j in article_to_id:
			inc_matrix[i][article_to_id[j]] = 1


# Euclidean distance aka sum of squared dimension distances
def distance(x,y):   
    return np.sqrt(np.sum((x-y)**2))

# Or just use this numpy function, should work about the same.
# np.linalg.norm(a-b)


# Make sorted list of best distances.
# This was for the non-SVD vectors. It didn't work quite right due to memory issues.
def best_scores(x):
	""" Input is an int corresponding to an article id. Compares the given article to each
	other article and outputs a ranked list of the most similar.
	"""
	scores = []
	for y in tqdm(range(len(inc_matrix))):
		dist = np.linalg.norm(inc_matrix[x]-inc_matrix[y])
		if dist < 20:
			scores.append((dist, id_to_article[y]))
			# had to stop early
			if y % 50000 == 0:
				return scores.sort(reverse=True)
	return scores.sort(reverse=True)


# Converting the matrix first into sci kit's sparse matrix...
s_inc_matrix = csr_matrix(inc_matrix)
# then using their svd with 300 dimensions; this is a hyperparameter.
svd = TruncatedSVD(n_components=300)
svd_inc_matrix = svd.fit_transform(s_inc_matrix)


# This is the version I used in the paper.
def svd_best_scores(x):
	""" Input is an int corresponding to an article id. Compares the given article to each
	other article and outputs a ranked list of the most similar.
	"""
	scores = []
	for y in tqdm(range(len(svd_inc_matrix))):
		# dist = np.linalg.norm(svd_inc_matrix[x]-svd_inc_matrix[y])
		dist = distance(svd_inc_matrix[x], svd_inc_matrix[y])
		scores.append((dist, id_to_article[y]))
	scores = sorted(scores, key=lambda x: x[0])
	return scores


########### Miscellaneous Section for references and other related approaches ###################



# This is for taking the incoming dictionary and setting it up in the right format for
# the node2vec algorithm. Format needs to be a text file with space-separated ints 
# (can be directed or undirected). Only one edge per line.
count = 0
with open('wiki.edgelist', 'w') as f:
	for i in incoming:
		for j in incoming[i]:
			if i.lower() in article_to_id and j.lower() in article_to_id:
				f.write(str(article_to_id[i.lower()]) + ' ' + str(article_to_id[j.lower()]))
				f.write('\n')
		count += 1
		if count % 100000 == 0:
			print(str(count) + ' articles parsed')


# This didn't make it into the paper as I had some problems with making it run correctly. Test results were not as good
# as SVD vectors.
def sim_measure(a, b):
	""" This is the equation for wikipedia-based link similarity from Milne and Witten (2008).
	It returns the similarity for two strings a and b, where the strings are the two articles of interest.
	It makes use of the list of links made TO each article.
	"""
	try:
		len_a = len(incoming_lower[a])
		len_b = len(incoming_lower[b])
		intersect = len([i for i in incoming_lower[a] if i in incoming_lower[b]])
		# print(intersect, len_a, len_b)
		num = math.log(max(len_a, len_b)) - math.log(intersect)
		denom = math.log(len(incoming_lower)) - math.log(min(len_a, len_b))
		return num / denom
	except KeyError:
		print('Argument not found in incoming')

# Making and ranking predictions using the above similarity measure.
def best_sims(x):
	scores = []
	for i in tqdm(incoming_lower):
		score = sim_measure(x, i)
		scores.append((score, i))
	scores = sorted(scores, key=lambda x:x[0])
	return scores

# reference list of xml tags for the Wiki dump.
# {http://www.mediawiki.org/xml/export-0.10/}page
# {http://www.mediawiki.org/xml/export-0.10/}title
# {http://www.mediawiki.org/xml/export-0.10/}ns
# {http://www.mediawiki.org/xml/export-0.10/}id
# {http://www.mediawiki.org/xml/export-0.10/}revision
# {http://www.mediawiki.org/xml/export-0.10/}id
# {http://www.mediawiki.org/xml/export-0.10/}parentid
# {http://www.mediawiki.org/xml/export-0.10/}timestamp
# {http://www.mediawiki.org/xml/export-0.10/}contributor
# {http://www.mediawiki.org/xml/export-0.10/}username
# {http://www.mediawiki.org/xml/export-0.10/}id
# {http://www.mediawiki.org/xml/export-0.10/}minor
# {http://www.mediawiki.org/xml/export-0.10/}model
# {http://www.mediawiki.org/xml/export-0.10/}format
# {http://www.mediawiki.org/xml/export-0.10/}text
# {http://www.mediawiki.org/xml/export-0.10/}sha1

# reference list of article + links example
# sample_links = ['Cai (state)|Cai',
#  'Jin (Chinese state)| Jin',
#  'File:Confuciustombqufu.jpg|thumb|200px|Tomb of Confucius in [[Cemetery of Confucius|Kong Lin cemetery',
#  'Qufu',
#  'Shandong Province',
#  'Zuo Zhuan',
#  'Ji Kangzi',
#  'disciples of Confucius|disciples',
#  'Five Classics',
#  'Ji Kangzi',
#  'Cemetery of Confucius|Kong Lin cemetery',
#  'Qufu',
#  'File:Dacheng Hall.JPG|thumb|The Dacheng Hall',
#  ' the main hall of the [[Temple of Confucius']