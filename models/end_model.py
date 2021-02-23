import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
import torch.utils.data as data

import sys 

class EndModel(nn.Module):
	'''
	Class representing a neural network trained on the distribution from the
	generative model
	'''

	def __init__(self, num_outputs):
		'''
		Constructor for an endmodel
		'''

		super(EndModel, self).__init__()
		self.dropout_rate = 0.5

		# setting up architecture for end model
		self.pretrained = resnet.resnet18(pretrained=True)
		for param in self.pretrained.parameters():
			param.requires_grad = False

		num_features = self.pretrained.fc.in_features
		self.pretrained.fc = nn.Linear(num_features, 49)
		self.out_layer = nn.Linear(49, num_outputs)
		self.dropout_layer = nn.Dropout(self.dropout_rate)

	def forward(self, images):
		'''
		Method for a foward pass of the endmodel
		'''

		pretrained_out = self.dropout_layer(F.relu(self.pretrained(images)))
		return self.out_layer(pretrained_out)

class OneLayer(nn.Module):
	'''
	Class representing a one layer neural network to be used as LR with soft labels
	on the votes matrix
	'''

	def __init__(self, num_inputs):
		super(OneLayer, self).__init__()
		self.out_layer = nn.Linear(num_inputs, 2)
		self.sigmoid = nn.Sigmoid()

	def forward(self, votes_matrix):
		return self.out_layer(votes_matrix)

def create_label_distribution(gen_model, wl_votes, new=False):
	'''
	Function to generate the label distribution for each training example to be
	used as labels in training the end model
	'''

	labels = []
	for i in wl_votes:
		label_dist = gen_model.get_label_distribution(i)
		labels.append(label_dist)
	labels = np.concatenate(labels, axis=0)
	# np.save("./data/bats/users/dsam/data/aa2/unseen_label_dists", labels)
	return labels

def generate_em_loader(gen_model, wl_votes, test_data, batch_size):
	'''
	Function to generate a trainloader for the end model
	'''

	xs = test_data
	ys = create_label_distribution(gen_model, wl_votes, new=True)

	# print(np.shape(xs))
	# print(np.shape(ys))

	em_loader = data.DataLoader(list(zip(xs, ys)), shuffle=True, batch_size=batch_size, num_workers=0)
	return em_loader

def create_endmodel():
	'''
	Function to create an end model object
	'''

	# train end model
	num_outputs = 2
	end_model = EndModel(num_outputs)
	return end_model

def create_one_layer(size):
	'''
	Wrapper function to create a One Layer (LR) model
	'''
	lr_model = OneLayer(size)
	return lr_model

def train_endmodel_soft(end_model, gen_model, wl_votes, test_data, device, batch_size=50, num_epochs=20, learning_rate=0.001):
	'''
	Function to train a end model
	'''
	
	num_data = wl_votes.shape[0]
	
	learn_params = []
	for name, param in end_model.named_parameters():
		if param.requires_grad:
			learn_params.append(param)

	optimizer = torch.optim.Adam(learn_params, lr=learning_rate)

	# creating end model train loader
	em_loader = generate_em_loader(gen_model, wl_votes, test_data, batch_size)
	
	end_model = end_model.to(device)
	# loss_func = nn.CrossEntropyLoss()
	loss_func = nn.KLDivLoss(reduction="batchmean")

	for ep in range(num_epochs):
		# print("Epoch : %d" % (ep))
		# b_count = 0

		for d in em_loader:

			x, y = d
			y = y.float()

			# zeroing gradient
			optimizer.zero_grad()

			# moving data to GPU/CPU
			inputs_batch = x.to(device)
			labels_batch = y.to(device)

			outputs = end_model(inputs_batch)
			probs = F.log_softmax(outputs, dim=1)
			# probs = F.softmax(outputs, dim=1)

			# loss = ce_loss(probs, labels_batch)
			# loss = loss_func(outputs, labels_batch)

			loss = F.kl_div(probs, labels_batch)
			loss.backward()
			optimizer.step()

			# b_count += 1
			# if b_count % 5 == 0:
			# 	print(loss.item())
	pass

def train_endmodel_hard(end_model, test_data, test_labels, device, batch_size=50, num_epochs=10, learning_rate=0.0001):
	'''
	Function to train a end model with hard labels
	'''
		
	learn_params = []
	for name, param in end_model.named_parameters():
		if param.requires_grad:
			learn_params.append(param)

	optimizer = torch.optim.Adam(learn_params, lr=learning_rate)

	# creating end model train loader
	em_loader = data.DataLoader(list(zip(test_data, test_labels)), shuffle=True, batch_size=batch_size, num_workers=0)
	
	end_model = end_model.to(device)
	loss_func = nn.CrossEntropyLoss()

	for ep in range(num_epochs):
		# print("Epoch : %d" % (ep))
		b_count = 0

		for d in em_loader:

			x, y = d
			# y = y.float()

			# zeroing gradient
			optimizer.zero_grad()

			# moving data to GPU/CPU
			inputs_batch = x.to(device)
			labels_batch = y.to(device)

			outputs = end_model(inputs_batch)
			# probs = F.softmax(outputs, dim=1)
			
			loss = loss_func(outputs, labels_batch)
			loss.backward()
			optimizer.step()

			# b_count += 1
			# if b_count % 4 == 0:
			# 	print(loss.item())
	pass

def load_endmodel(num_outputs, path):
	'''
	Function to load an end model
	'''

	endmodel = EndModel(num_outputs)	      
	endmodel.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
	return endmodel

def ce_loss(outputs, prob_dist):
	'''
	Method to compute the KL divergence for label model distributions
	'''

	# \sum P(x) log ( P(x) / Q(x) )
	return (prob_dist * (prob_dist / outputs).log()).sum(dim=1).mean()

def bce_loss(outputs, prob_dist):
	'''
	Method to compute the KL divergence for label model distributions
	'''

	# \sum P(x) log ( P(x) / Q(x) )
	bce = prob_dist * (prob_dist / outputs).log() + (1 - prob_dist) * ((1 - prob_dist) / outputs).log() 
	return bce.mean()

def eval_em(test_data, test_labels, end_model, device): 
	'''
	Function to evaluate the end model 
	'''

	dl = data.DataLoader(list(zip(test_data, test_labels)), shuffle=True, batch_size=100, num_workers=0)
	learned_preds = []
	ls = []
	
	end_model = end_model.to(device)

	for d in dl:

		x, y = d

		# moving data to GPU/CPU
		inputs_batch = x.to(device)
		outputs = end_model(inputs_batch)
		_, end_preds = torch.max(outputs, 1)
		end_preds = end_preds.cpu().detach().numpy()
		y = y.detach().numpy()
		learned_preds.append(end_preds)
		ls.append(y)

	learned_preds = np.concatenate(learned_preds)
	learned_preds = learned_preds + 1
	ls = np.concatenate(ls)

	# print(list(learned_preds))
	# print(list(ls))

	print("End Model Accuracy: %f" % (np.mean(learned_preds == ls)))
	return np.mean(learned_preds == ls)
