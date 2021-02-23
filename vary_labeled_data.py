import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision.models import resnet
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit

import os
import sys
import argparse
import distutils
import pickle
import itertools
import csv

import data.aa2_data as AA2
import data.attributes as attr
import models.end_model as EM
import models.weak_labeler as WL
import models.gen_model as GM
import models.all_algo as ALL
import models.lp as LP
import models.heuristic_algo as HA

# setting random seeds
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
cuda0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def str2bool(v):
	'''
	Used to help argparse library 
	'''
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

def compute_differences(votes_matrix, indices):
	'''
	Function to compute the differences between weak labelers (for Alessio)
	'''

	n = len(indices)
	D = np.zeros((n, n))
	for index_i, i in enumerate(indices):
		for index_j, j in enumerate(indices):
			votes_i = votes_matrix[:, index_i]
			votes_j = votes_matrix[:, index_j]
			D[index_i, index_j] = 1 - np.mean(votes_i == votes_j)
	return D

def compute_epsilons(votes_matrix, indices, labels):
	'''
	Function to compute the individual errors of weak labelers
	'''

	n = len(indices)
	epsilons = np.zeros(n)
	for index, i in enumerate(indices):
		votes_i = votes_matrix[:, index]
		epsilons[index] = 1 - np.mean(votes_i == labels)
	return epsilons 

def correct_epsilons(epsilons, votes):
	'''
	Function to flip votes and errors when error > 0.5
	'''

	for i, e in enumerate(epsilons):
		if e > 0.5:
			epsilons[i] = 1 - e
			votes[:, i] = np.where(votes[:, i] == 1, 2, 1)
	return epsilons, votes

def run_all(classes, task, features, ratio):
	'''
	Function to evaluate ALL
	'''

	signals, test_data, test_labels, _ = GM.create_signal_matrix(classes, features)
	bounds = np.load("stats/" + str(ratio) + "/epsilons" + task + ".npy")

	# inverting signals from bad weak signals
	for i, err in enumerate(bounds):
		if err > 0.5:
			signals[:,i] = 1 - signals[:,i]
			bounds[i] = 1 - bounds[i]
	
	print("All")		
	print("Task " + task)
		
	train_all_em(test_data, signals.T, test_labels, bounds, task)

def get_votes(classes, features, names):
	'''
	Function to get a set of votes
	'''

	num_examples = len(names)
	num_wls = len(features)

	# constructing list of weak labelers to load/invert
	indices = []
	for feature in features:
		# checking for negation
		to_add = [True, 0]
		if feature[0] == "!":
			to_add[0] = False
		to_add[1] = attr.attributes.index(feature.replace("!", ""))
		indices.append(to_add)

	votes_matrix = np.zeros((num_examples, num_wls))

	for i, tup in enumerate(indices):

		vote_correct, index = tup
		vote_dict = pickle.load(open("./data/votes/wl_votes_%d.p" % (index), "rb"))
		
		for j, name in enumerate(names):
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
				
	return votes_matrix

def get_signals(classes, features, names):
	'''
	Function to get a set of signals
	'''

	num_examples = len(names)
	num_wls = len(features)

	# constructing list of weak labelers to load/invert
	indices = []
	for feature in features:
		# checking for negation
		to_add = [True, 0]
		if feature[0] == "!":
			to_add[0] = False
		to_add[1] = attr.attributes.index(feature.replace("!", ""))
		indices.append(to_add)

	sig_matrix = np.zeros((num_examples, num_wls))
	for i, tup in enumerate(indices):

		sig_correct, index = tup
		sig_dict = pickle.load(open("./data/signals/signals_%d.p" % (index), "rb"))
		for j, name in enumerate(names):
			sig = sig_dict[name]
			if sig_correct:
				sig = 1 - sig
			sig_matrix[j][i] = sig 

	return sig_matrix

def correct_epsilons(epsilons, votes):
	'''
	Function to flip votes and errors when error > 0.5
	'''

	for i, e in enumerate(epsilons):
		if e > 0.5:
			epsilons[i] = 1 - e
			votes[:, i] = np.where(votes[:, i] == 1, 2, 1)

	return epsilons, votes

def correct_signals(epsilons, signals):
	'''
	Function to flip votes and errors when error > 0.5
	'''

	for i, e in enumerate(epsilons):
		if e > 0.5:
			epsilons[i] = 1 - e
			signals[:, i] = 1 - signals[:, 1]

	return epsilons, signals

if __name__ == "__main__":

	# setting up argparsers
	parser = argparse.ArgumentParser()
	parser.add_argument('--start', default=1, type=int, help="start integer for creating votes")
	parser.add_argument('--mv', default=False, type=str2bool, help="for evaluating majority vote without flipping")
	parser.add_argument('--ssl_ds', default=False, type=str2bool, help="run script to evaluate Dawid Skene w/ SSL")
	parser.add_argument('--lp', default=False, type=str2bool, help="running using lp/closed formula")
	parser.add_argument('--all', default=False, type=str2bool, help="run script to evaluate all training")
	parser.add_argument('--lr', default=0.0001, type=float, help="learning rate")
	parser.add_argument('--ha', default=False, type=str2bool, help="for evaluating heuristic algorithm")
	parser.add_argument('--ha_algo', default=1, type=int, help="which heuristic algo to use")

	args = parser.parse_args()

	ratios = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 0.9]

	# getting task information defined by start
	unseen_classes = AA2.get_test_classes()
	combs = list(itertools.combinations(range(10), 2))
	classes = combs[args.start - 1]
	unseen = [unseen_classes[classes[0]], unseen_classes[classes[1]]]
	task = str(classes[0]) + str(classes[1])
	features = attr.get_feature_diffs(unseen)

	print("Task: " + task)
	(train_data, train_labels, train_names), (test_data, test_labels, test_names) = AA2.gen_unseen_data_split(classes, seed)
	
	# run task at a time for remaining things
	for r, ratio in enumerate(ratios):
		print("Ratio: %f" % (ratio))

		# checking for edge case of task 47 erroring bc not enough data
		if task == "47" and ratio == 0.01:
			continue

		sss = StratifiedShuffleSplit(n_splits=3, test_size=1 - ratio, random_state=seed)
		split = 0

		for l_index, ul_index in sss.split(train_data, train_labels):
			print("Split: %d" % split)

			# gathering labeled and unlabeled data
			l_data, l_labels, l_names = train_data[l_index], train_labels[l_index], train_names[l_index]
			ul_data, ul_labels, ul_names = train_data[ul_index], train_labels[ul_index], train_names[ul_index]
			l_votes = get_votes(classes, features, l_names)
			ul_votes = get_votes(classes, features, ul_names)
			test_votes = get_votes(classes, features, test_names)
			error_estimates = compute_epsilons(l_votes, range(np.shape(l_votes)[1]), l_labels)

			# running MV baselines
			if args.mv:
				test_votes = get_votes(classes, features, test_names)
				gm = GM.create_gen_model(2, np.shape(test_votes)[1])
				mv = gm.get_most_probable_labels(test_votes)
				error_estimates, flipped_votes = correct_epsilons(error_estimates, test_votes)
				mv_flip = gm.get_most_probable_labels(flipped_votes)

				GM.train_gen_model(gm, test_votes)
				print("MV: %f" % (np.mean(mv == test_labels)))
				print("MV Flip %f" % (np.mean(mv_flip == test_labels)))

			# running semi-supervised DS baseline
			if args.ssl_ds:
				gen_model = GM.create_ssl_gen_model(2, np.shape(l_votes)[1])
				votes = np.concatenate((l_votes, test_votes))
				labels = np.concatenate((l_labels, np.zeros(test_votes.shape[0]))).astype(int)
				GM.train_ssl_gen_model(gen_model, votes, labels)
				ds = GM.get_ssl_predictions(gen_model, test_votes)
				ssl_ds_acc = np.mean(ds == test_labels)
				print("SSL DS: %f" % (ssl_ds_acc))

			# running Adversarial Label Learning
			if args.all:		
				test_signals = get_signals(classes, features, test_names)
				all_acc = ALL.eval_all_lr(test_votes, test_signals.T, test_labels, test_votes, test_labels, error_estimates)
				print("ALL Accuracy: " + str(all_acc))

			# flipping votes
			if args.lp or args.ha:
				test_votes = get_votes(classes, features, test_names)
				error_estimates, test_votes = correct_epsilons(error_estimates, test_votes)

			# running our Linear program approach with the closed bound
			if args.lp:				
				new_votes, idx, comb_dict, epsilons = LP.iterative_algo(test_votes, error_estimates)
				new_votes += 1
				lp_sub = comb_dict[np.shape(test_votes)[1]]
				algo_out = comb_dict[idx]
				print("LP Subset: " + str(lp_sub))
				print("Bound: %f" % (epsilons[np.shape(test_votes)[1]]))
				lp_votes = new_votes[:, lp_sub]

				gm = GM.create_gen_model(2, np.shape(test_votes)[1])
				mv = gm.get_most_probable_labels(test_votes)
				gm_lp = GM.create_gen_model(2, np.shape(lp_votes)[1])
				mv_lp = gm_lp.get_most_probable_labels(lp_votes)
				print("LP MV: %f" % (np.mean(mv_lp == test_labels)))

			if args.ha:
				if args.ha_algo == 1:
					best_sub, best_ep = HA.heuristic_algo1(1, error_estimates, test_votes - 1, 5, 9)
				elif args.ha_algo == 2:
					best_sub, best_ep = HA.heuristic_algo1(2, error_estimates, test_votes - 1, 5, 9)
				print("Heuristic Algorithm Subset: " + str(best_sub))
				
				ha_votes = test_votes[:, best_sub]
				gm = GM.create_gen_model(2, np.shape(ha_votes)[1])
				ha_mv = gm.get_most_probable_labels(ha_votes)
				print("HA MV: %f" % (np.mean(ha_mv == test_labels)))
				print("Bound: %f" % (best_ep))