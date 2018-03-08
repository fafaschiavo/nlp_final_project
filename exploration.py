import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle

print '------------ Lets explore the data ------------'

data_folder = 'data/'

edges_df = pd.read_csv(data_folder + 'training_set.txt', sep = ' ', names = ["nodeID1", "nodeID2", "is_linked"])
nodes_df = pd.read_csv(data_folder + 'node_information.csv', names = ["nodeID", "year", "title", "author", "journal", "abstract"])

# # ============= GET LONGEST AND SHORTEST ABSTRACT
# max_length = 0
# min_length = 1000
# for index, row in nodes_df.iterrows():
# 	if len(row['abstract']) > max_length:
# 		max_length = len(row['abstract'])
# 	if len(row['abstract']) < min_length:
# 		min_length = len(row['abstract'])

# print max_length #2130
# print min_length #18

# # ============= CREATE DATASET CORPORA BY SETTING TOGETHER UNIQUE WORDS AND FILTERING STOP WORDS
# # ============= FINISH BY SAVING THE CORPORA FOR FURTHER USE
total_words_corpora = []
for index, row in nodes_df.iterrows():
	print index
	total_words_corpora = total_words_corpora + word_tokenize(row['abstract'])


total_set_corpora = set(total_words_corpora)

stop_words = set(stopwords.words('english'))
total_filtered_corpora = [w for w in total_words_corpora if not w in stop_words]
filtered_corpora = [w for w in total_set_corpora if not w in stop_words]

stemmer = nltk.stem.PorterStemmer()
stemmed_corpora = [stemmer.stem(w) for w in filtered_corpora]
stemmed_corpora = set(stemmed_corpora)

print 'Total words in all abstracts:'
print len(total_words_corpora)
print 'Total words in all abstracts with filtered stopwords:'
print len(total_filtered_corpora)
print 'Total unique words in abstracts:'
print len(total_set_corpora)
print 'Total unique words with filtered stopwords:'
print len(filtered_corpora)

print 'Now saving corpora...'
with open(data_folder + 'corpora.pickle', 'wb') as file:
    pickle.dump(stemmed_corpora, file)






# # ============= USEFUL FUNCTIONS
# first_abstract = nodes_df['abstract'][0]
# tokens_abstract = word_tokenize(first_abstract)

# stop_words = set(stopwords.words('english'))
# filtered_abstract = [w for w in tokens_abstract if not w in stop_words]