import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision.models import resnet
import torchvision.transforms as transforms

import os
import sys
import argparse
import distutils
import pickle
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import data.aa2_data as AA2
import data.attributes as attr

import models.weak_labeler as WL
import models.gen_model as GM
import models.all_algo as ALL
import models.lp as LP
import models.heuristic_algo as HA

cuda0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
np.set_printoptions(threshold=sys.maxsize)

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
	
def eval_mv(error_estimates, votes, labels):
	'''
	Function to train a generative model (Dawid Skene)
	'''
	
	gen_model = GM.create_gen_model(2, np.shape(votes)[1])
	mv = gen_model.get_most_probable_labels(votes)
	
	flip_votes = np.zeros_like(votes)
	for i, err in enumerate(error_estimates):
		if err > 0.5:
			flip_votes[:,i] = np.where(votes[:,i] == 1, 2, 1)
		else:
			flip_votes[:,i] = votes[:,i]	
	mv_flip = gen_model.get_most_probable_labels(flip_votes)
	mv_acc = np.mean(mv == labels)
	mv_flip_acc = np.mean(mv_flip == labels)
	print("MV: %f" % (mv_acc))
	print("MV Flip: %f" % (mv_flip_acc))

	return mv_acc, mv_flip_acc

def train_ssl_gm(l_votes, l_labels, ul_votes, ul_labels):
	'''
	Function to train a SSL generative model (Dawid Skene)
	'''

	gen_model = GM.create_ssl_gen_model(2, np.shape(l_votes)[1])
	votes = np.concatenate((l_votes, ul_votes))
	labels = np.concatenate((l_labels, np.zeros_like(ul_labels)))
	GM.train_ssl_gen_model(gen_model, votes, labels)
	ds = GM.get_ssl_predictions(gen_model, ul_votes)
	ssl_ds_acc = np.mean(ds == ul_labels)
	print("SSL DS: %f" % (ssl_ds_acc))
	return ssl_ds_acc

def eval_lp(error_estimates, test_votes, test_labels):
	'''
	Function to evaluate original LP
	'''

	new_votes, idx, comb_dict, eps = LP.iterative_algo(test_votes, error_estimates)
	comb = comb_dict[np.shape(test_votes)[1]]

	print(eps)
	print(comb)

	lp_votes = test_votes[:, comb]
	gm = GM.create_gen_model(2, np.shape(lp_votes)[1])
	lp_mv = gm.get_most_probable_labels(lp_votes)

	print("LP MV: %f" % (np.mean(lp_mv == test_labels)))
	print("Bound: %f" % (eps[np.shape(test_votes)[1]]))
	return np.mean(lp_mv == test_labels), eps[np.shape(test_votes)[1]]

def eval_ha(error_estimates, test_votes, test_labels, algo=1, size=5):
	'''
	Function to evaluate the heuristic algo
	'''

	# flipping votes
	error_estimates, test_votes = correct_epsilons(error_estimates, test_votes)
	print("Test: " + str(np.shape(test_votes)))	

	votes_matrix = np.where(test_votes == 2, 1, 0)

	# running heuristic algorithm
	best_sub, best_ep = HA.heuristic_algo1(algo, error_estimates, votes_matrix, size, 9)

	print("HA: " + str(best_sub))
	ha_votes = test_votes[:, best_sub]

	gm = GM.create_gen_model(2, np.shape(ha_votes)[1])
	ha_mv = gm.get_most_probable_labels(ha_votes)

	print("HA MV: %f" % (np.mean(ha_mv == test_labels)))
	print("Bound: %f" % (best_ep))
	return best_sub, np.mean(ha_mv == test_labels), best_ep

if __name__ == "__main__":

	# setting up argparsers
	parser = argparse.ArgumentParser()
	parser.add_argument('--start', default=0, type=int, help="start integer for creating votes")
	parser.add_argument('--evalmv', default=False, type=str2bool, help="run script to evaluate majority vote")
	parser.add_argument('--ssl_ds', default=False, type=str2bool, help="run script to evaluate Semi-Supervised DS")
	parser.add_argument('--lp', default=False, type=str2bool, help="running using lp/closed formula")
	parser.add_argument('--all', default=False, type=str2bool, help="run script to evaluate all training")
	parser.add_argument('--lr', default=0.0001, type=float, help="learning rate")
	parser.add_argument('--ha', default=False, type=str2bool, help="run script to evaluate heuristic algorithm")
	parser.add_argument('--ha_algo', default=1, type=int, help="run script to evaluate heuristic algorithm")

	args = parser.parse_args()
	lr_string = np.format_float_positional(np.float32(args.lr))
	unseen_classes = AA2.get_test_classes()

	seeds = [0, 1, 2, 3, 4]

	combs = list(itertools.combinations(range(10), 2))
	classes = combs[args.start - 1]
	unseen = [unseen_classes[classes[0]], unseen_classes[classes[1]]]
	task = str(classes[0]) + str(classes[1])
	features = attr.get_feature_diffs(unseen)

	# getting data
	votes, data, labels, names1 = GM.create_votes_matrix(classes, features )
	signals, _, _, _ = GM.create_signal_matrix(classes, features)
	print("Loaded Data")

	# 5 random seeds
	for seed in seeds:

		# setting random seeds
		print("Seed %d" % (seed))
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)

		# splitting data
		train_indices, test_indices = train_test_split(range(len(labels)), test_size=0.5, random_state=seed, stratify=labels)
		train_votes, train_data, train_labels = votes[train_indices], data[train_indices], labels[train_indices]
		test_votes, test_data, test_labels = votes[test_indices], data[test_indices], labels[test_indices]
		test_signals = signals[test_indices]			
		error_estimates = compute_epsilons(train_votes, range(np.shape(train_votes)[1]), train_labels)

		# evaluating majority vote, ds, majority vote with flips
		if args.evalmv:
			mv_acc, mv_flip_acc = eval_mv(error_estimates, test_votes, test_labels)			
		
		if args.ssl_ds:
			ssl_ds_acc = train_ssl_gm(train_votes, train_labels, test_votes, test_labels)

		elif args.lp:
			lp_acc, lp_bound = eval_lp(error_estimates, test_votes, test_labels)

		elif args.all:			
			all_acc = ALL.eval_all_lr(test_votes, test_signals.T, test_labels, test_votes, test_labels, error_estimates)

		elif args.ha:
			# min size for heuristic algorithm
			size = 5
			print("Heuristic Algorithm")
			_, ha_acc, best_ep = eval_ha(error_estimates, test_votes, test_labels, args.ha_algo, size)