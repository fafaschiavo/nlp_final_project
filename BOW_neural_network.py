import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle
from random import choice, randint

import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical, pad_sequences




print '------------ Lets try a neural network with BOW data ------------'

testing_set = 1000
links_fraction = 0.5
data_folder = 'data/'
models_folder = 'models/'

number_of_epochs = 5

edges_df = pd.read_csv(data_folder + 'training_set.txt', sep = ' ', names = ["nodeID1", "nodeID2", "is_linked"])
testing_edges_df = pd.read_csv(data_folder + 'training_set.txt', sep = ' ', names = ["nodeID1", "nodeID2", "is_linked"])
nodes_df = pd.read_csv(data_folder + 'node_information.csv', names = ["nodeID", "year", "title", "author", "journal", "abstract"])

with open(data_folder + 'corpora.pickle', 'rb') as file:
	corpora = pickle.load(file)

with open(data_folder + 'bow_dictionary.pickle', 'rb') as file:
	BOW_dict = pickle.load(file)



# # Creates random edges
# edges_df = pd.DataFrame(columns=["nodeID1", "nodeID2", "is_linked"])
# for x in xrange(0,100):
# 	random_1 = choice(BOW_dict.keys())
# 	random_2 = choice(BOW_dict.keys())
# 	is_linked = randint(0, 1)
# 	edges_df.loc[x] = [random_1, random_2, is_linked]




train_X = []
train_Y = []
max_mixture_size = 0
for index, row in edges_df.sample(frac=links_fraction).iterrows():
	first_paper = BOW_dict[row['nodeID1']]
	second_paper = BOW_dict[row['nodeID2']]
	current_mixture = first_paper + second_paper
	if len(current_mixture) > max_mixture_size:
		max_mixture_size = len(current_mixture)

	train_X.append(current_mixture)
	train_Y.append(int(row['is_linked']))

train_X = pad_sequences(train_X)
train_Y = to_categorical(train_Y, 2)

test_X = train_X[:testing_set]
train_X = train_X[testing_set:]

test_Y = train_Y[:testing_set]
train_Y = train_Y[testing_set:]

# Building convolutional network
network = input_data(shape=[None, max_mixture_size], name='input')
network = tflearn.embedding(network, input_dim=len(corpora), output_dim=128)
branch1 = conv_1d(network, 128, 3, padding='valid', activation='relu', regularizer="L2")
branch2 = conv_1d(network, 128, 4, padding='valid', activation='relu', regularizer="L2")
branch3 = conv_1d(network, 128, 5, padding='valid', activation='relu', regularizer="L2")
network = merge([branch1, branch2, branch3], mode='concat', axis=1)
network = tf.expand_dims(network, 2)
network = global_max_pool(network)
network = dropout(network, 0.5)
network = fully_connected(network, 2, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.001,
                     loss='categorical_crossentropy', name='target')
# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit(train_X, train_Y, n_epoch = number_of_epochs, shuffle=True, validation_set=(test_X, test_Y), show_metric=True, batch_size=64)


model.save(models_folder + 'bow_concat_epochs' + str(number_of_epochs) + '.tflearn')








# for index, row in nodes_df.iterrows():
# 	print index
# 	print row['abstract']