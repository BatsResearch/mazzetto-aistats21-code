import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision.models import resnet
import torchvision.transforms as transforms

import data.attributes as attr
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

def generate_weak_labeler():
	'''
	Function to generate a single weak labelers (ie. attribute detectors)
	'''

	model = resnet.resnet18(pretrained=True)
	for param in model.parameters():
		param.requires_grad = False

	num_features = model.fc.in_features
	model.fc = nn.Linear(num_features, 2)
	
	return model

def train_weak_labeler(model, index, trainloader, valloader, testloader, device, lr=0.0001, batch_size=40, num_epochs=1):
	'''
	Function to train weak labelers to detect a given attribute

	Args:
	model - the weak labeler to train
	index - the index of the attribute
	trainloader - the train dataset for the weak labeler
	valloader - the validation dataset for the weak labeler
	device - the device to train the weak labeler on
	'''

	feature = attr.attributes[index]
	print("Training model %d for %s on: " % (index, feature) + str(device))

	# computing ratio of true training classes to not true training classes
	train_ratio = attr.compute_train_ratio(feature)

	print("Train ratio: %f" % (train_ratio))
	if train_ratio == 1 or train_ratio == 0:
		loss_weight = torch.Tensor([1, 1])
	else:
		loss_weight = torch.Tensor([1 / (1 - train_ratio), 1 / train_ratio])
	loss_weight = loss_weight.to(device)
	
	learn_params = []
	for name, param in model.named_parameters():
		if param.requires_grad:
			learn_params.append(param)

	# changed on 6/23
	optimizer = torch.optim.Adam(learn_params, lr=lr)
	objective_function = torch.nn.CrossEntropyLoss(weight=loss_weight)	
	model.to(device)

	# looping for epochs
	for ep in range(num_epochs):

		total_ex = 0
		pos_ex = 0

		for x, attribute, y, _ in trainloader:

			# zeroing gradient
			optimizer.zero_grad()
			inputs = x
			labels = attribute[:, index]

			# moving data to GPU/CPU
			inputs = inputs.to(device)
			labels = labels.to(device)

			pos_ex += torch.sum(labels).item()
			total_ex += batch_size

			outputs = model(inputs)
			loss = objective_function(outputs, labels)
			loss.backward()
			optimizer.step()

		# evaluating on validation set
		val_acc = []
		v_pos_corr = 0
		v_pos_total = 0
		for x, attribute, y, _ in valloader:

			# convert to feature labels rather than class labels
			inputs = x
			labels = attribute[:, index]

			np_labels = labels.numpy()

			# moving data to GPU/CPU
			inputs = inputs.to(device)
			labels = labels.to(device)

			outputs = model(inputs)
			_, preds = torch.max(outputs, 1)

			np_preds = torch.Tensor.cpu(preds).numpy()
			v_pos_total += torch.sum(labels).item()

			for i in np_labels:
				if i == 1:
					 v_pos_corr += np_preds[i]

			acc = torch.sum(preds == labels).item() / x.size()[0]
			val_acc.append(acc)

		print("Epoch: %d" % (ep))
		print("Validation Accuracy: %f" % (np.mean(val_acc)))
		print("\n")

	print("Class Ratio for attribute %s: %f" % (feature, pos_ex / total_ex))
	pass
		

def evaluate_wl(wl, index, testloader, device): 
	'''
	Function to evaluate a weak labeler on the test data

	Args:
	wl - the weak labeler
	index - the index of the attribute the weak labeler detects
	testloader - the test data to evaluate on
	device - the device to run the model on 
	'''
	
	feature = attr.attributes[index]
	print("Evaluating weak labeler " + str(index) + " detecting: " + feature)
	accuracies = []

	predictions = []
	labs = []
	count = 0

	cm = 0

	for x, attribute, y, _ in testloader:
		inputs = x
		labels = attribute[:, index]

		np_labels = labels.numpy()
		
		# moving data to GPU/CPU
		inputs = inputs.to(device)
		labels = labels.to(device)

		_, preds = torch.max(wl(inputs), 1)
		acc = torch.sum(preds == labels).item() / x.size()[0]
		accuracies.append(acc)
		np_preds = torch.Tensor.cpu(preds).numpy()

		predictions.append(np_preds)
		labs.append(np_labels)

		# computing confusion matrix
		cm += confusion_matrix(np_labels, np_preds, labels=[0, 1])

	print("Test Accuracy: %f" % (np.mean(accuracies)))
	print(cm)

	predictions = np.concatenate(predictions)
	labs = np.concatenate(labs)

	cm2 = confusion_matrix(labs, predictions, labels=[0,1])	
	print(cm2)
	
	return np.mean(accuracies)

def load_wl(index):
	'''
	Function to load an individual weak labeler
	'''

	model = resnet.resnet18()
	num_features = model.fc.in_features
	model.fc = nn.Linear(num_features, 2)  
	wl_path = "./data/weak_labelers/wl_" + str(index)
	model.load_state_dict(torch.load(wl_path, map_location=torch.device("cpu")))
	return model

