import pulp as lp
import numpy as np
import itertools
import math
import operator as op
from statistics import mean

def gen_bstring2(n, a):
    if(n==0):
        return [a.copy()]
    else:
        a[n-1]=0
        l1 = gen_bstring2(n-1, a)
        a[n-1]=1
        l2 = gen_bstring2(n-1, a)
        return l1+l2

def gen_bstring(n):
    return gen_bstring2(n,np.zeros(n,dtype=np.int8))

def v_to_s(a):
    s = ""
    for x in a:
        s = s + str(x)
    return s

def flip_string(s):
    output = ""
    for char in s:
        if char == "0":
            output = output + "1"
        else:
            output = output + "0"
    return output 

def majority_vote(vec): 
    if np.sum(vec) > len(vec)/2: 
        return 1
    else: 
        return 0

def build_vec_from_mat(idx, difference):
    ret = []
    for i in range(len(idx)):
        for j in range(i+1,len(idx)):
            ret.append(difference[idx[i]][idx[j]])
    return np.array(ret)

def build_vec(idx, epsilon):
    ret = []
    for i in range(len(idx)):
        ret.append(epsilon[idx[i]])
    return ret

def get_diff_from_votes(vote_data):
    L = len(vote_data[0])
    output = [[0 for i in range(L)] for j in range(L)]
    for a in range(len(vote_data)):
        for i in range(L):
            for j in range(L):
                if vote_data[a][i] != vote_data[a][j]:
                    output[i][j] += 1
    for i in range(L):
        for j in range(L):
            output[i][j] = float(output[i][j])/len(vote_data)
    
    return output

# Given n indices , prints a dict s.t., if the n labelers are ordered 
# by index, then the key is the sequence, and the val is the prob. 
# For example, the value in dict["011"] for 3 labelers 
# is the probability of seeing the sequence 0,1,1.
def build_probs_from_votes(indices, votes):
    N = len(indices)
    possible = gen_bstring(N)
    adict = {}
    for a in possible:
        if (a[0] == 0):
            adict[v_to_s(a)] = 0
    for i in range(len(votes)):
        binary = ""
        tmp = []
        for j in indices:
            binary += str(int(votes[i][j]))
            tmp.append(votes[i][j])
        if (votes[i][0] != 0):
            adict[flip_string(binary)] = adict.get(flip_string(binary),0) + 1
        else:
            adict[binary] = adict.get(binary,0) + 1
    s = len(votes)
    for key in adict:
        adict[key] = float(adict[key])/s
    return adict

def compute_upper_bound_old(data):  # pair( epsilon, matrix)
    eps = data[0] # errors of labelers
    diff = data[1] # differences
    N = len(eps) # num labelers
    possible_a = gen_bstring(N)
    p_a = [lp.LpVariable("p_"+v_to_s(a), 0, 1) for a in possible_a]
    q_a = [lp.LpVariable("q_"+v_to_s(a), 0, 1) for a in possible_a]

    prob = lp.LpProblem("myProblem",lp.LpMaximize)
    # Objective function
    prob += lp.lpSum( [q_a[j] for j in range(len(possible_a))])


    # Constraints
    prob += lp.lpSum([q for q in q_a] + [p for p in p_a]) == 1
    for a in range(len(p_a)):
        prob+= p_a[a] >= 0
        prob+= q_a[a] >= 0

    # Constraint 1
    for i in range(N):
        prob += lp.lpSum( [p_a[j] for j in range(len(possible_a)) if possible_a[j][i] != majority_vote(possible_a[j])] + [q_a[j] for j in range(len(possible_a)) if possible_a[j][i] == majority_vote(possible_a[j])]) == eps[i]

    # Constraint 2
    x = 0
    for i in range(N):
        for j in range(N):
            if(i < j):
                prob+=lp.lpSum([p_a[k] for k in range(len(possible_a)) if possible_a[k][i] != possible_a[k][j] ]+[q_a[k] for k in range(len(possible_a)) if possible_a[k][i] != possible_a[k][j] ]) == diff[x]
                x = x+1

    status = prob.solve()
    #print(lp.LpStatus[prob.status])
    if lp.LpStatus[prob.status] == "Optimal":
        return prob.objective.value()
    else:
        return 2

def fft_metric(ind,epsilon,difference,k):
    eps = build_vec(ind,epsilon)
    diffs = build_vec_from_mat(ind,difference)
    L = len(ind)
    #print(eps)
    #print(diffs)
    output = mean(eps) - k*mean(diffs)
    #print(output)
    return output

def compute_upper_bound_new(data):  # pair( epsilon, matrix)
    eps = data[0] # errors of labelers
    vote_probs = data[1] # differences
    N = len(eps) # num labelers
    possible_a = gen_bstring(N)
    possible_a_str = []
    for s in possible_a:
        possible_a_str.append(v_to_s(s))
    p_a = [lp.LpVariable("p_"+v_to_s(a), 0, 1) for a in possible_a]
    q_a = [lp.LpVariable("q_"+v_to_s(a), 0, 1) for a in possible_a]
    # t_a = [lp.LpVariable("t_"+v_to_s(a), 0, 1) for a in possible_a]

    prob = lp.LpProblem("myProblem",lp.LpMaximize)
    # Objective function
    prob += lp.lpSum( [q_a[j] for j in range(len(possible_a))])

    # Constraints 3 and 4 
    prob += lp.lpSum([q for q in q_a] + [p for p in p_a]) == 1
    for a in range(len(p_a)):
        prob+= p_a[a] >= 0
        prob+= q_a[a] >= 0

    # Constraint 1
    for i in range(N):
        prob += lp.lpSum( [p_a[j] for j in range(len(possible_a)) if possible_a[j][i] != majority_vote(possible_a[j])] + [q_a[j] for j in range(len(possible_a)) if possible_a[j][i] == majority_vote(possible_a[j])]) == eps[i]

    # Constraint 2
    seen = []
    for j in range(len(possible_a)):
        if possible_a_str[j] not in seen:
            ind = possible_a_str.index(flip_string(possible_a_str[j]))
            if majority_vote(possible_a[j]):
                prob += lp.lpSum(p_a[j] + q_a[j] + p_a[ind] + q_a[ind]) == (vote_probs[possible_a_str[ind]])
            else:
                prob += lp.lpSum(p_a[j] + q_a[j] + p_a[ind] + q_a[ind]) == (vote_probs[possible_a_str[j]])
            seen.append(possible_a_str[j])
            seen.append(possible_a_str[ind])
    
    #print(prob)
    status = prob.solve()
    if lp.LpStatus[prob.status] == "Optimal":
        return prob.objective.value()
    else:
        return 2 # LP Terminated without finding optimal solution

def compute_upper_bound_new2(data):  # pair( epsilon, matrix)
    eps = data[0] # errors of labelers
    p = data[1] # differences
    N = len(eps) # num labelers
    possible = [a for a in gen_bstring(N) if (a[0] == 0)]
    #print(possible)
    c_a = [lp.LpVariable("c_"+v_to_s(a), 0, 1) for a in possible]
    # t_a = [lp.LpVariable("t_"+v_to_s(a), 0, 1) for a in possible_a]

    prob = lp.LpProblem("myProblem",lp.LpMaximize)
    # Objective function
    prob += lp.lpSum([p[v_to_s(possible[j])]*c_a[j] for j in range(len(possible))])

    # Constraint 1
    for i in range(N):
        prob += lp.lpSum([p[v_to_s(possible[j])]*c_a[j] for j in range(len(possible)) if possible[j][i] == majority_vote(possible[j])]+[p[v_to_s(possible[j])]*(1-c_a[j]) for j in range(len(possible)) if possible[j][i] != majority_vote(possible[j])]) <= eps[i] 
    
    #print(prob)
    status = prob.solve()
    #prob.solve(PULP_CBC_CMD(msg=0))
    if lp.LpStatus[prob.status] == "Optimal":
        return prob.objective.value()
    else:
        return 2 # LP Terminated without finding optimal solution

def heuristic_algo1(algo, epsilon, votes_matrix, min_labelers, size):
    
    votes = votes_matrix
    # print(votes)
    difference = get_diff_from_votes(votes)

    # Correct the labelers so that they all have error rates < 1/2
    for i in range(len(epsilon)):
        if(epsilon[i]>1/2):
            for k in range(len(votes)):
                votes[k][i] = 1 - votes[k][i]
            for j in range(len(epsilon)):
                if j!=i:        
                    difference[i][j] = 1 - difference[i][j]
                    difference[j][i] = 1 - difference[j][i]
            epsilon[i] = 1 - epsilon[i]
    # print(epsilon)
    if algo == 1:
        data = difference 
    else:
        data = votes

    labelers = list(range(len(epsilon)))
    best_ep = 1
    best_S = 0

    for j in range(len(labelers)): 
        print("ON LABELER")
        print(j)

        S = [j]
        tmp_ep = 2

        # add best two labelers until we reach minimum number of labelers
        while len(S) < min_labelers:
            tmp_ep = 2
            add_S = []
            unused = labelers.copy()

            for x in S: 
                unused.remove(x)
            
            combs = list(itertools.combinations(unused,2))
            for c in combs:
                tmp = S.copy()
                for ind in range(len(c)):
                    tmp.append(c[ind])
                tmp.sort()
                if algo == 1:
                    x = compute_upper_bound_old((build_vec(tmp, epsilon),build_vec_from_mat(tmp, data)))
                else:
                    x = compute_upper_bound_new2((build_vec(tmp,epsilon),build_probs_from_votes(tmp,data)))

                if x < tmp_ep:
                    tmp_ep = x
                    add_S = c

            if len(add_S) == 0:
                break
            else:
                for ind in range(len(add_S)):
                    S.append(add_S[ind])
                S.sort()
        # skipping if no optimal solution is found
        if len(S) < min_labelers:
            print("skipping")
            continue

        previous_ep = tmp_ep
        curr_ep = tmp_ep
        
        while True: 
            if len(S) == size:
                if curr_ep < best_ep:
                    best_ep = curr_ep
                    best_S = S
                break
                
            add_S = 0
            unused = labelers.copy()
            for x in S:
                unused.remove(x)
            if( len(unused) < 2): break
            combs = list(itertools.combinations(unused, 2))
            for c in combs:
                tmp = S.copy()
                for ind in range(len(c)):
                    tmp.append(c[ind])
                tmp.sort()

                if algo == 1:
                    x = compute_upper_bound_old((build_vec(tmp, epsilon),build_vec_from_mat(tmp, data)))
                else:
                    x = compute_upper_bound_new2((build_vec(tmp,epsilon),build_probs_from_votes(tmp,data)))
                if x < curr_ep:
                    curr_ep = x
                    add_S = c     
            if curr_ep < previous_ep:
                previous_ep = curr_ep
                for ind in range(len(add_S)):
                    S.append(add_S[ind])
                S.sort()
            else:
                if curr_ep < best_ep:
                    best_ep = curr_ep
                    best_S = S
                break
    return(best_S,best_ep)


#num is the number of labelers needed, delta is the scaling param
#in the algorithm 
def heuristic_algo2(epsilon, votes_matrix, num, delta):
    # epsilon = np.load(epsilon_file)
    votes = votes_matrix - 1
    # votes = np.load(votes_file) - 1 # -1 based on file format
    difference = get_diff_from_votes(votes)

    # Correct the labelers so that they all have error rates < 1/2
    for i in range(len(epsilon)):
        if(epsilon[i]>1/2):
            for k in range(len(votes)):
                votes[k][i] = 1 - votes[k][i]
            for j in range(len(epsilon)):
                if j!=i:
                    difference[i][j] = 1 - difference[i][j]
                    difference[j][i] = 1 - difference[j][i]
            epsilon[i] = 1 - epsilon[i]

    L = len(epsilon)
    combs = list(itertools.combinations(range(L), num))
    min_dist = 99999
    best_c = (0,0,0)
    for c in combs:
        x = fft_metric(c,epsilon,difference,delta)
        if x < min_dist:
            #print(best_c)
            min_dist = x
            best_c = c
    return(min_dist,best_c)
