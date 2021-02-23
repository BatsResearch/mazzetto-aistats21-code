import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision.models import resnet
import torchvision.transforms as transforms
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

import os
import sys
import argparse
import distutils
import pickle

import data.aa2_data as AA2
import models.weak_labeler as WL
import models.gen_model as GM
import data.attributes as attr

# getting attributes matrix and list
attributes = AA2.get_attributes()
attributes_matrix = AA2.create_attribute_matrix()
train_classes, val_classes, test_classes = AA2.get_train_classes(), AA2.get_val_classes(), AA2.get_test_classes()
classes_map = AA2.get_classes()

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

def train_wl(index, lr=0.0001):
	'''
	Function to create and train a weak labeler on a given attribute given by the
	corresponding index

	Args:
	index - the index of the attribute to learn to detect
	'''

	batch_size = 50
	num_epochs = 5

	trainloader = AA2.get_trainloader(batch_size)
	valloader = AA2.get_valloader(batch_size)
	testloader  = AA2.get_testloader(batch_size)
	
	print("Loaded data")
	print("Learning rate %f" % (lr))

	# creating weak labelers for difference in binary class attributes (49 total)
	wl = WL.generate_weak_labeler()
	WL.train_weak_labeler(wl, index, trainloader, valloader, testloader, cuda0, lr=lr, batch_size=batch_size, num_epochs=num_epochs)
	WL.evaluate_wl(wl, index, testloader, cuda0)

	# saving model
	torch.save(wl.state_dict(), "data/weak_labelers/wl_%d" % (index))
	print("Saved model: %d" % (index))

def compute_f1(indices, create=False, signals=False, split=0):
	'''
	Method to compute the f1 score for a subset of weak labelers

	Args:
	indices: the indices of which weak labelers to evaluate
	'''

	testloader = AA2.get_testloader(50, shuffle=True)
	device = torch.device("cpu")
	print("Loaded data")
	for index in indices:
		if create:
			path = "data/votes/wl_votes_%d.p" % (index)
			if os.path.exists(path):
				print("Votes for index %d already exist" % (index))
				continue
			else:
				print("Votes for index %d do not exist" % (index))

		# loading weak labeler
		wl = WL.load_wl(index)

		true_labels = []
		predictions = []
		names_list = []
		signal_list = []

		for x, attribute, y, names in testloader:

			labels = attribute[:, index]

			# moving data to device
			inputs = x.to(device)
			labels = labels.to(device)
			outputs = wl(inputs)
			_, preds = torch.max(outputs, 1)
			
			# computing weak signal probabilities
			if signals:
				sig = F.softmax(outputs, dim=1)
				sig = sig[:,1]
				sig = sig.detach().numpy()
				signal_list.append(sig) 

			preds = torch.Tensor.cpu(preds).numpy()
			labels = labels.detach().numpy()
			y_pred = preds
			true_labels.append(labels)
			predictions.append(y_pred)
			names_list.append(names)

		true_labels = np.concatenate(true_labels)
		predictions = np.concatenate(predictions)
		names_list = np.concatenate(names_list)

		votes_dict = {}
		signal_dict = {}

		if signals:
			signal_list = np.concatenate(signal_list)

		for i, name in enumerate(names_list):
			votes_dict[name] = predictions[i]
			if signals:
				signal_dict[name] = signal_list[i]

		if create:
			pickle.dump(votes_dict, open("data/votes/votes_%d.p" % (index), "wb"))
		if signals:
			pickle.dump(signal_dict, open("data/signals/signals_%d.p" % (index), "wb"))
		
		else:
			if np.sum(true_labels) == 0 or np.sum(true_labels) == len(true_labels):
				f1 = -1
			else:
				f1 = f1_score(true_labels, predictions, labels=[0, 1])
			
			# printing accuracy/f1 statistics
			print("Classifier %d, feature %s" % (index, attr.attributes[index]))
			print("Class Balance : %f" % (np.sum(true_labels) / len(true_labels)))
			print("Accuracy : %f , F1 Score : %f" % (np.mean(true_labels == predictions), f1))
			print(confusion_matrix(true_labels, predictions, labels=[0, 1]))

def create_votes_matrix_names(names):
	'''
	Function to create a votes matrix from a list of datapoint names
	'''

	votes_matrix = np.zeros((len(names), 85))
	for wl in range(85):
		votes_dict = pickle.load(open("data/votes/wl_votes_%d.p" % (wl), "rb"))
		for i, n in enumerate(names):
			votes_matrix[i,wl] = votes_dict[n]
	return votes_matrix

def create_signals_matrix_names(names):
	'''
	Function to create a signals matrix from a list of datapoint names
	'''

	sig_matrix = np.zeros((len(names), 85))
	for wl in range(85):
		sig_dict = pickle.load(open("data/signals/signals_%d.p" % (wl), "rb"))
		for i, n in enumerate(names):
			sig_matrix[i,wl] = sig_dict[n]
	return sig_matrix

def compute_wl_accuracies():
	'''
	Function to compute the accuracies of each weak labeler
	'''
	accuracies = np.zeros((85,))
	name_order = pickle.load(open("data/unseen_names.p", "rb"))
	attributes = np.load("data/unseen_attributes.npy")
	for i in range(85):
		votes_dict = pickle.load(open("data/votes/wl_votes_%d.p" % (i), "rb"))
		votes = [votes_dict[x] for x in name_order]
		acc = np.mean(votes == attributes[:, i])
		accuracies[i] = acc
	return accuracies

if __name__ == "__main__":
	
	# setting up argparsers
	parser = argparse.ArgumentParser()
	parser.add_argument('--train', default=False, type=str2bool, help="run to train weak labelers")
	parser.add_argument('--lr', default=0.0001, type=float, help="learning rate")
	parser.add_argument('--start', default=-1, type=int, help="index to start training labelers")
	parser.add_argument('--create', default=False, type=str2bool, help="create weak labelers votes on unseen data")
	parser.add_argument('--create_signals', default=False, type=str2bool, help="create weak labelers probs on unseen data")

	args = parser.parse_args()
	
	# can change to train a particular set of weak labelers
	indices = range(85)
	if args.train:
		# generate unseen dataset
		if not os.path.isfile("data/unseen_data.npy"):
			print("Converting unseen class data into numpy matrices")
			AA2.gen_unseen_dataset()
			print("Finished saving data")

		for i in indices:
			if not os.path.exists("data/weak_labelers/wl_%d" % (i)):
				train_wl(i, lr=args.lr)
			else:
				print("Trained weights for wl %d already exist" % (i))
	else:
		compute_f1(indices, create=args.create, signals=args.create_signals)



