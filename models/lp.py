import pulp as lp
import numpy as np
import itertools
import argparse
import pickle

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

def compute_difference(votes,i,j):
	'''
	Function t compute the individaul difference between two labelers
	'''

	n = len(votes)
	
	cum = 0

	for v in votes:
		if v[i] != v[j]:
			cum += 1

	return cum / n

def get_eps_and_diff(eps, diff, i,j,k):

	err = [eps[i], eps[j], eps[k]]
	d = []

	d.append(diff[max(i,j)][min(i,j)])
	d.append(diff[max(i,k)][min(i,k)])
	d.append(diff[max(k,j)][min(k,j)])
	return err,d

def compute_upper_bound(eps,diff,i,j,k):
	
	e,d = get_eps_and_diff(eps,diff,i,j,k) #12 13 23
	q = 1/2*max([e[0]+e[1]-d[2]-d[1], e[1]+e[2]-d[0]-d[1], e[0]+e[2]-d[0]-d[2],0])
	ubb = e[0]+e[1]+e[2]-1/2*d[0]-1/2*d[1]-1/2*d[2]-2*q
	
	# fix issues with negative bounds
	if ubb < 0:
		return 0
	else:
		return ubb
	
	# return max(0,ubb)

def majority_vote(v, nl):
	
	x = 0
	div = len(nl)
	for i in nl:
		x += v[i]
	if x/div < 1/2:
		return 0
	else:
		return 1

def greedy_algo(votes, eps):
	'''
	Function to iteratively select the best 3 wls to produce a weighted vote (Alessio's algorithm)
	'''
	
	# Get all labelers to have error rate < 0.5 
	for i in range(len(eps)):
		if eps[i] > 0.5:
			eps[i] = 1 - eps[i]
			for v in votes:
				v[i] = 1 - v[i]
	
	current_labelers = []
	for i in range(len(eps)):
		current_labelers.append([i])

	# Precompute differencies
	D = [[]]

	for i in range(1,len(eps)):
		d = []
		for j in range(i):
			d.append(compute_difference(votes,i,j))     
		D.append(d)

	# Implementation of the algorithm
	continue_iterate = True

	while continue_iterate:
		combs = list(itertools.combinations(range(len(current_labelers)), 3))
		best_c = None
		best_upb = min(eps)
		for c in combs:
			i = c[0]
			j = c[1]
			k = c[2]

			upb = compute_upper_bound(eps, D, i, j, k)
			# print(best_upb)
			if(upb < 0.99 * best_upb and (upb < min(eps[i], eps[j], eps[k]))):
				best_c = c
				best_upb = upb
		
		if best_c != None:

			# print(best_c)
			# print(best_upb)
			i = best_c[0]
			j = best_c[1]
			k = best_c[2]
			
			# Add the new labeler
			new_labeler = [i,j,k]
			current_labelers.append(new_labeler)
			

			# Update the error rates
			eps = np.append(eps,best_upb)

			# Update the vote matrix
			new_votes =[]
			for i in range(len(votes)):
				res = majority_vote(votes[i],new_labeler)
				v = np.append(votes[i], res)
				new_votes.append(v)
			votes = new_votes
			
			# Update the differencies
			d = []

			for j in range(len(eps)-1):
				d.append(compute_difference(votes,len(eps)-1,j))     
			D.append(d)
		else:
			continue_iterate = False
	
	return np.array(votes)[:, -1]

def iterative_algo(test_votes, eps):
	'''
	Function to iteratively select the best 3 wls to produce a weighted vote (Alessio's algorithm)
	'''

	# fix votes
	votes = test_votes - 1

	comb_dict = {}
	init_length = len(eps)

	# # Get all labelers to have error rate < 0.5 
	# for i in range(len(eps)):
	# 	if eps[i] > 0.5:
	# 		eps[i] = 1 - eps[i]
	# 		for v in votes:
	# 			v[i] = 1 - v[i]
	
	current_labelers = []
	for i in range(len(eps)):
		current_labelers.append([i])

	# Precompute differencies
	D = [[]]

	for i in range(1,len(eps)):
		d = []
		for j in range(i):
			d.append(compute_difference(votes,i,j))     
		D.append(d)

	# Implementation of the algorithm
	continue_iterate = True
	used = [False]*len(eps)

	while continue_iterate:
		combs = list(itertools.combinations(range(len(current_labelers)), 3))
		best_c = None
		best_upb = 1
		for c in combs:
			i = c[0]
			j = c[1]
			k = c[2]
			upb = compute_upper_bound(eps,D,i,j,k)
			
			if((used[i] == False and used[j] == False and used[k] == False) and (upb < best_upb*0.99)):
				best_c = c
				best_upb = upb
		
		if best_c != None:
			
			# print(best_c)
			# print(best_upb)
			
			i = best_c[0]
			j = best_c[1]
			k = best_c[2]

			# Update used
			used[i] = True
			used[j] = True
			used[k] = True
			
			used.append(False)
			
			# Add the new labeler
			new_labeler = [i,j,k]
			current_labelers.append(new_labeler)
			
			comb_dict[len(current_labelers) - 1] = best_c

			# Update the error rates
			eps = np.append(eps,best_upb)

			# Update the vote matrix
			new_votes =[]
			for i in range(len(votes)):
				res = majority_vote(votes[i],new_labeler)
				v = np.append(votes[i],res)
				new_votes.append(v)
			votes = new_votes
			
			# Update the differencies
			d = []
			for j in range(len(eps)-1):
				d.append(compute_difference(votes,len(eps)-1,j))     
			D.append(d)
		else:
			continue_iterate = False
		
	idx = np.argmin(eps[init_length:]) + init_length
	return np.array(votes), idx, comb_dict, eps

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--size', default=3, type=int, help="size of weak labeler combinations to consider")
	args = parser.parse_args()
	
	opt_dict1 = {}
	opt_dict2 = {}

	ind = True
	split = 0

	task_combs = list(itertools.combinations(range(10), 2))
	
	for task in task_combs:
		task_sub = str(task[0]) + str(task[1])
		epsilon = np.load("stats/ind_%d/epsilons" % (split) + task_sub + ".npy")
		difference = np.load("stats/ind_%d/differences" % (split) + task_sub + ".npy")
		# epsilon = correct_epsilons(epsilon)
		
		L = len(epsilon)
		print("Task: " + task_sub)

		# LP1
		combs = list(itertools.combinations(range(L), args.size))
		best_score = 1
		best_c = None
		for c in combs:
			x = compute_upper_bound((build_vec(c),build_vec_from_mat(c)))
			# print(x)
			if x < best_score:
				best_score = x
				best_c = c
		print("LP1: " + str(best_c))
		opt_dict1[task_sub] = best_c

		# LP2
		combs = list(itertools.combinations(range(L), 3))
		best_score2 = 1
		best_c2 = (0,0,0)
		for c in combs:
			x = compute_upper_bound3((build_vec(c),build_vec_from_mat(c)))
			if x < best_score2:
				best_score2 = x
				best_c2 = c

		print("LP2: " + str(best_c2))
		opt_dict2[task_sub] = best_c2

	pickle.dump(opt_dict1, open("lp1_subsets_ind_%d.p" % (split), "wb"))
	pickle.dump(opt_dict2, open("lp2_subsets_ind_%d.p" % (split), "wb"))
	# 3: (7, 37, 46)
	# 4: (6, 7, 37, 46)
	



