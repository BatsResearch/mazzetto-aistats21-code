import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import data.aa2_data as AA2

# getting attributes matrix and list
attributes = AA2.get_attributes()
attributes_matrix = AA2.create_attribute_matrix()
train_classes, val_classes, test_classes = AA2.get_train_classes(), AA2.get_val_classes(), AA2.get_test_classes()
classes_map = AA2.get_classes()

def get_feature_diffs(unseen_classes):
	'''
	Function to generate the feature differences between two unseen classes

	Args:
	unseen_classes - a list of two strings that represent the unseen class name
	'''

	binary_class_indices = [classes_map[unseen_classes[0]], classes_map[unseen_classes[1]]]
	binary_class_features = []

	atts_1 = attributes_matrix[binary_class_indices[0]]
	atts_2 = attributes_matrix[binary_class_indices[1]]
	for i in range(85):
		if atts_1[i] != atts_2[i]:
			if atts_1[i] == 1:
				binary_class_features.append(attributes[i])
			else:
				binary_class_features.append("!" + attributes[i])
	return binary_class_features

def create_feature_label(train_attributes, feature):
	'''
	Function to generate a label for a specific index of a weak labeler
	'''

	list_f = feature.split("+")
	label = None

	for l_f in list_f:
		# checking for negation
		if l_f.startswith("!"):
			f_index = attributes.index(l_f[1:])
			if train_attributes[f_index] == 0 and (label == None or label == 1):
				label = 1
			else:
				label = 0
		else:
			f_index = attributes.index(l_f)
			if train_attributes[f_index] == 1 and (label == None or label == 1):
				label = 1
			else:
				label = 0

	return label

def compute_train_ratio(feature):
	'''
	Method to check the presence of the given attribute in the training classes
	'''
	feats = feature.split("+")
	num_classes = 0
	total_classes = len(train_classes)

	for train_c in train_classes:
		class_attributes = attributes_matrix[classes_map[train_c]]
		sat = True	
		for f in feats:

			# checking for number of classes with attribute
			if f.startswith("!"):
				f_ind = attributes.index(f[1:])
				if class_attributes[f_ind] == 0:
					pass
				else:
					sat = False
					break
			else:
				f_ind = attributes.index(f)
				if class_attributes[f_ind] == 1:
					pass
				else:
					sat = False
					break
		if sat:
			num_classes += 1
	return num_classes / total_classes
