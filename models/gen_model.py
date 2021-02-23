import numpy as np
import random
import torch

import sys
import argparse
import distutils
import pickle
from scipy import sparse

import data.aa2_data as AA2
import data.attributes as attr
import models.labelmodels.labelmodels.naive_bayes as NB
import models.labelmodels.labelmodels.semi_supervised as SS

np.set_printoptions(threshold=sys.maxsize)

def convert_to_labels(np_labels, class_list):
	'''
	Function to convert from an array of test labels to labels in class_list
	'''

	return np.vectorize(lambda x: class_list.index(x) + 1)(np_labels)

def wl_to_binary_votes(features):
	'''
	Function to convert from feature array of weak labelers to votes to 
	each class
	'''

	votes = []
	for i in range(0, len(features), 2): # looping by two
		if features[i + 1] >= features[i]:
			votes.append(1)
		else:
			votes.append(2)
	
	return np.array(votes)

def accuracy(label_dist, labels):
	'''
	Compute accuracy of the predictions of a model

	Args:
	label_dist - the output of a generative model
	labels - the true labels of the data
	'''
	return np.mean(np.argmax(label_dist, axis=1) == labels)

def create_gen_model(num_outputs, num_voters):
	'''
	Function to create a Dawid Skene generative model

	Args:
	num_outputs - the number of possible output classes
	num_voters - the number of voters given as input
	'''

	# creating gen to train on distributions
	gen_model = NB.NaiveBayes(num_outputs, num_voters, init_acc=0.80)
	return gen_model

def create_ssl_gen_model(num_outputs, num_voters):
	'''
	Function to create a SSL Dawid Skene generative model

	Args:
	num_outputs - the number of possible output classes
	num_voters - the number of voters given as input
	'''

	gen_model = SS.SemiSupervisedNaiveBayes(num_outputs, num_voters)
	return gen_model

def train_ssl_gen_model(ssl_gen_model, votes, labels):
	'''
	Function to train a generative model in a SSL fashion
	'''

	ssl_gen_model.estimate_label_model(votes, labels)


def get_ssl_predictions(gen_model, votes):
	"""Returns the predictions of the semi-supervised generative model"""
	votes = sparse.csr_matrix(votes, dtype=np.int)
	labels = np.ndarray((votes.shape[0], gen_model.num_classes))
	
	batch_size = 4096
	batches = [(sparse.coo_matrix(
		votes[i * batch_size: (i + 1) * batch_size, :],
		copy=True),)
		for i in range(int(np.ceil(votes.shape[0] / batch_size)))
	]

	offset = 0
	for votes, in batches:
		class_balance = gen_model._get_norm_class_balance()
		lf_likelihood = gen_model._get_labeling_function_likelihoods(votes)
		jll = class_balance + lf_likelihood
		for i in range(votes.shape[0]):
			p = torch.exp(jll[i, :] - torch.max(jll[i, :]))
			p = p / p.sum()
			for j in range(gen_model.num_classes):
				labels[offset + i, j] = p[j]
		offset += votes.shape[0]

	return np.argmax(labels, axis=1) + 1

def train_gen_model(gen_model, data):
	'''
	Function to train a generative model

	Args:
	gen_model - the model to train
	data - the data which to train the generative model on (estimate accuracies)
	'''

	gen_model.estimate_label_model(data)
	pass

def create_wl_matrix():
	'''
	Function to generate the votes matrix from weak labelers

	Args:
	features - the features of the weak labelers to create votes for
	load - if to load already existing dataset

	Returns:
	creates a votes matrix on datapoints in the order of the names ordering
	'''

	# constructing list of weak labelers to load/invert
	test_labels = np.load("data/unseen_labels.npy")
	names = pickle.load(open("data/unseen_names.p", "rb"))
	votes_matrix = np.zeros((len(names), 85))

	for i in range(85):
		vote_dict = pickle.load(open("data/votes/wl_votes_%d.p" % (i), "rb"))

		for j, name in enumerate(names):
			votes_matrix[j][i] = vote_dict[name] 
				
	return votes_matrix, test_labels, names

def create_votes_matrix(classes, features, load=False, ind=False, train=False, split=0):
	'''
	Function to generate the votes matrix on unseen data for a given set of weak labelers

	Args:
	features - the features of the weak labelers to create votes for
	load - if to load already existing dataset

	Returns:
	creates a votes matrix on datapoints in the order of the name_order
	'''

	# constructing list of weak labelers to load/invert
	indices = []
	for feature in features:
		# checking for negation
		to_add = [True, 0]
		if feature[0] == "!":
			to_add[0] = False
		to_add[1] = attr.attributes.index(feature.replace("!", ""))
		indices.append(to_add)

	path = "data/"
	test_data = np.load(path + "unseen_data.npy")
	test_labels = np.load(path + "unseen_labels.npy")
	names = pickle.load(open(path + "unseen_names.p", "rb"))
	valid_indices = np.concatenate([np.nonzero(test_labels == classes[0]), 
									np.nonzero(test_labels == classes[1])], axis=None) 

	valid_data = test_data[valid_indices]
	valid_labels = test_labels[valid_indices]
	valid_labels = convert_to_labels(valid_labels, classes)
	name_order = [names[i] for i in valid_indices]
	num_examples = len(name_order)
	num_wls = len(features)

	# print(num_examples, num_wls)

	votes_matrix = np.zeros((num_examples, num_wls))

	for i, tup in enumerate(indices):

		vote_correct, index = tup
		vote_dict = pickle.load(open("./data/votes/wl_votes_%d.p" % (index), "rb"))
	
		for j, name in enumerate(name_order):
			to_assign = vote_dict[name]
			
			if to_assign == 1 and vote_correct:
				to_assign = 1
			elif to_assign == 1 and not vote_correct:
				to_assign = 2
			elif to_assign == 0 and vote_correct:
				to_assign = 2
			else:
				to_assign = 1
			
			votes_matrix[j][i] = to_assign 
				
	return votes_matrix, valid_data, valid_labels, name_order

def create_signal_matrix(classes, features, load=False, ind=False, split=0):
	'''
	Function to generate the signals matrix on unseen data for a given set of weak labelers

	Args:
	features - the features of the weak labelers to create votes for

	Returns:
	creates a signals matrix on datapoints in the order of the name_order
	'''

	# constructing list of weak labelers to load/invert
	indices = []

	for feature in features:
		# checking for negation
		to_add = [True, 0]
		if feature[0] == "!":
			to_add[0] = False
		to_add[1] = attr.attributes.index(feature.replace("!", ""))
		indices.append(to_add)

	path = "data/"
	test_data = np.load(path + "unseen_data.npy")
	test_labels = np.load(path + "unseen_labels.npy")
	names = pickle.load(open(path + "unseen_names.p", "rb"))

	valid_indices = np.concatenate([np.nonzero(test_labels == classes[0]), 
									np.nonzero(test_labels == classes[1])], axis=None) 

	valid_data = test_data[valid_indices]
	valid_labels = test_labels[valid_indices]
	valid_labels = convert_to_labels(valid_labels, classes)
	name_order = [names[i] for i in valid_indices]

	num_examples = len(name_order)
	num_wls = len(features)

	sig_matrix = np.zeros((num_examples, num_wls))

	for i, tup in enumerate(indices):

		sig_correct, index = tup
		sig_dict = pickle.load(open("data/signals/signals_%d.p" % (index), "rb"))

		for j, name in enumerate(name_order):
			sig = sig_dict[name]
			if sig_correct:
				sig = 1 - sig
			sig_matrix[j][i] = sig 
	return sig_matrix, valid_data, valid_labels, name_order
