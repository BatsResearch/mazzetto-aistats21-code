import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Subset

from sklearn.model_selection import train_test_split

import os
import pickle
from glob import glob
from PIL import Image

# defining filepaths
data_path = "data/Animals_with_Attributes2/"
classes_path = "data/classes.txt"
test_path 	= "data/testclasses.txt"
train_path 	= "data/trainclasses1.txt"
val_path	= "data/valclasses1.txt"
images_path = data_path + "JPEGImages"
attributes_path = data_path + "predicates.txt"

class AA2_Dataset(data.dataset.Dataset):
	'''
	Class for the Animals with Attributes 2 Dataset
	'''

	def __init__(self, classes, transform=None):
		self.class_map = get_classes()
		self.predicate_binary_mat = create_attribute_matrix()

		# looping through directory to get 
		self.image_names = []
		self.image_indices = []
		self.transform = transform
		self.image_labels = []

		for c in classes:
			FOLDER_DIR = os.path.join(images_path, c)
			file_descriptor = os.path.join(FOLDER_DIR, '*.jpg')
			files = glob(file_descriptor)

			for file_name in files:
				self.image_names.append(file_name)
				self.image_indices.append(self.class_map[c])
				self.image_labels.append(classes.index(c))
	
	def __len__(self):
		return len(self.image_names)

	def __getitem__(self, index):
		'''
		Function to get an image
		'''

		im = Image.open(self.image_names[index])
		im_copy = im
		
		if im.getbands()[0] == 'L':
			im_copy = im_copy.convert('RGB')
		
		if self.transform:
			im_copy = self.transform(im_copy)
		im_array = np.array(im_copy)
		im_index = self.image_indices[index]
		im_predicate = self.predicate_binary_mat[im_index,:]
		im.close()

		return im_array, im_predicate, self.image_labels[index], self.image_names[index]

def get_trainloader(batch_size, shuffle=True):
	'''
	Function to get a trainloader for the AA2 dataset
	'''

	classes = get_train_classes()

	transform = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
	])

	train_dataset = AA2_Dataset(classes, transform=transform)
	trainloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
	print("train length: " + str(len(trainloader)))
	return trainloader

def get_valloader(batch_size, shuffle=True):
	'''
	Function to get a valloader for the AA2 dataset
	'''

	classes = get_val_classes()

	transform = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
	])

	val_dataset = AA2_Dataset(classes, transform=transform)
	valloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
	print("val length: " + str(len(valloader)))
	return valloader

def get_testloader(batch_size, shuffle=False):
	'''
	Function to get a testloader for the AA2 dataset
	'''

	classes = get_test_classes()

	transform = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
	])

	test_dataset = AA2_Dataset(classes, transform=transform)
	testloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
	return testloader

def get_train_classes():
	'''
	Function to get the train classes
	'''
	train_classes = []
	with open(train_path) as train_f:
		train_classes = [x.strip() for x in train_f.readlines()]
	return train_classes

def get_val_classes():
	'''
	Function to get the train classes
	'''
	val_classes = []
	with open(val_path) as val_f:
		val_classes = [x.strip() for x in val_f.readlines()]
	return val_classes

def get_test_classes():
	'''
	Function to get the test classes
	'''
	test_classes = []
	with open(test_path) as test_f:
		test_classes = [x.strip() for x in test_f.readlines()]
	return test_classes

def get_classes():
	'''
	Function to create a mapping from class name to index
	'''
	class_to_index = {}
	with open(classes_path) as f:
		index = 0
		for line in f:
			class_name = line.split('\t')[1].strip()
			class_to_index[class_name] = index
			index += 1
	
	return class_to_index

def get_attributes():
	'''
	Function to get the list of attributes of the AA2 dataset
	'''

	attributes = []
	with open(attributes_path) as a_f:
		for line in a_f.readlines():
			attribute = line.strip().split()[1]
			attributes.append(attribute)
	return attributes

def create_attribute_matrix():
	'''
	Function to create a matrix of classes and attributes
	'''

	predicate_binary_mat = np.array(np.genfromtxt("data/predicate-matrix-binary.txt", dtype='int'))
	return predicate_binary_mat

def gen_unseen_dataset():
	'''
	Function to generate the unseen dataset (data, labels, and image names)
	'''

	testloader = get_testloader(100, shuffle=False)
	unseen_data = []
	attributes = []
	labels = []
	names = []

	total = len(testloader)
	count = 0

	for x, attr, y, name in testloader:
		unseen_data.append(x)
		attributes.append(attr)
		labels.append(y)
		names += name

		if count % 10 == 0:
			print("%d out of %d batches completed" % (count, total))
		count += 1
	
	unseen_data = np.concatenate(unseen_data)
	attributes = np.concatenate(attributes)
	labels = np.concatenate(labels)

	# gives an ordering to data and converts to numpy or pickle files
	np.save("data/unseen_data", unseen_data)
	np.save("data/unseen_attributes", attributes)
	np.save("data/unseen_labels", labels)
	pickle.dump(names, open("data/unseen_names.p", "wb"))

def gen_unseen_data_split(classes, seed):
	'''
	Creating unseen dataset with a train and test split for an inductive approach
	'''

	base_path = "data/" 
	unseen_data = np.load(base_path + "unseen_data.npy")
	unseen_attributes = np.load(base_path + "unseen_attributes.npy")
	unseen_labels = np.load(base_path + "unseen_labels.npy")
	unseen_names = np.array(pickle.load(open(base_path + "unseen_names.p", "rb")))

	valid_indices = np.concatenate([np.nonzero(unseen_labels == classes[0]), 
									np.nonzero(unseen_labels == classes[1])], axis=None) 

	unseen_data = unseen_data[valid_indices]
	unseen_labels = unseen_labels[valid_indices]
	unseen_attributes = unseen_attributes[valid_indices]
	unseen_names = unseen_names[valid_indices]

	# converting labels
	unseen_labels = np.where(unseen_labels == classes[0], 1, 2)

	# splitting to test and train data (w/ stratified)
	train_indices, test_indices = train_test_split(range(len(unseen_labels)), test_size=0.5, random_state=seed, stratify=unseen_labels)

	train_data, train_labels = unseen_data[train_indices], unseen_labels[train_indices]
	train_atts, train_names = unseen_attributes[train_indices], unseen_names[train_indices]

	test_data, test_labels = unseen_data[test_indices], unseen_labels[test_indices]
	test_atts, test_names = unseen_attributes[test_indices], unseen_names[test_indices]

	# splitting train data into labeled and unlabeled
	return (train_data, train_labels, train_names), (test_data, test_labels, test_names)