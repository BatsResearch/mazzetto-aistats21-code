import numpy as np 
import itertools
import pickle

import data.aa2_data as AA2
import data.attributes as attr
import models.gen_model as GM

def commitee_potential(accuracies):
	'''
	Function to compute the committee potential of a set of labelers
	'''
	return np.sum((accuracies - 0.5) * np.log(accuracies / (1 - accuracies)))

if __name__ == "__main__":

	unseen_classes = AA2.get_test_classes()
	combs = list(itertools.combinations(range(10), 2))
	cp_map  = {}

	for classes in combs: 

		unseen = [unseen_classes[classes[0]], unseen_classes[classes[1]]]
		features = attr.get_feature_diffs(unseen)
		votes, _, labels, _ = GM.create_votes_matrix(classes, features)

		accuracies = np.array([np.mean(votes[:, i] == labels) for i in range(votes.shape[1])])
		# print(accuracies)
		cp = commitee_potential(accuracies)
		# print(cp)
		cp_map[classes] = cp
	
	print(cp_map)
	# pickle.dump(cp_map, open("cps.p", "wb"))