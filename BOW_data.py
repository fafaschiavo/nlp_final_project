import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle

def convert_to_BOW_array(content, corpora):
	stop_words = set(stopwords.words('english'))
	tokenized_content = word_tokenize(content)

	stemmer = nltk.stem.PorterStemmer()

	filtered_content = [stemmer.stem(w) for w in tokenized_content if not w in stop_words]
	filtered_content = set(filtered_content)

	BOW_array = []
	for word in filtered_content:
		corpora_index = list(corpora).index(word)
		BOW_array.append(corpora_index)

	return BOW_array

print '------------ Lets convert to a BOW array ------------'

data_folder = 'data/'
edges_df = pd.read_csv(data_folder + 'training_set.txt', sep = ' ', names = ["nodeID1", "nodeID2", "is_linked"])
nodes_df = pd.read_csv(data_folder + 'node_information.csv', names = ["nodeID", "year", "title", "author", "journal", "abstract"])

with open(data_folder + 'corpora.pickle', 'rb') as file:
	filtered_corpora = pickle.load(file)

nodes_dict = {}
for index, row in nodes_df.iterrows():
	print index
	BOW_array = convert_to_BOW_array(row['abstract'], filtered_corpora)
	nodes_dict[row['nodeID']] = BOW_array

print 'Now saving data...'
with open(data_folder + 'bow_dictionary.pickle', 'wb') as file:
	pickle.dump(nodes_dict, file)
