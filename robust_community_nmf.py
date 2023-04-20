### Community Detection Algorithm

import numpy as np
from scipy import sparse

## Read input from file
FILEPATH = "zacharys_karate_club"
# FILEPATH = "dolphins_social_network"
# FILEPATH = "les_miserables"
train_i = []
train_j = []
train_val = []

with open(FILEPATH, "r") as file:
    num_nodes, num_edges, num_comms = [int(val) for val in next(file).split()]
    for line in file:
        i, j = line.split()
        train_i.append(int(i))
        train_j.append(int(j))
        train_val.append(1)

M = sparse.coo_matrix((train_val + train_val, (train_i + train_j, train_j + train_i)), shape=(num_nodes, num_nodes)).tocsr()

"""
Optimization algorithm of EA2NMF
Input:
    The adjacency matrix A;
    The set of nodes V;
    The hyper-parameter λ;
    The number of communities k;
    The numbers of iterations initer, outiter;
Output: the set of communities S = {s1, s2, · · · , sk};
"""

def soft_threshold(S, _lambda):
    S = np.where(np.logical_and(S >= -_lambda/2, S <= _lambda/2), 0, S)
    S = np.where(S < -_lambda/2, S + _lambda/2, S)
    S = np.where(S > _lambda/2, S - _lambda/2, S)
    return S

## Initialize hyperparameters
n = num_nodes
k = num_comms

outiter = 1000
_lambda = 0.3

## Initialize trainable parameters
W = np.random.rand(n,k)
H = np.random.rand(k,n)
S = np.random.rand(n,n)
X = A = M.toarray()

## Initialize W and H
for _ in range(1):
    W, H = W * (X @ H.T) / (W @ H @ H.T), H * (W.T @ X) / (W.T @ W @ H)

for _ in range(outiter):
    ## Update S perturbation
    S = X - W @ H
    S = soft_threshold(S, _lambda)
    
    ## Update W, with H fixed
    num = np.abs((S-X) @ H.T) - ((S-X) @ H.T)
    denom = 2*(W @ H @ H.T)
    W = num / denom * W

    ## Update H, with W fixed
    num = np.abs(W.T @ (S-X)) - (W.T @ (S-X))
    denom = 2*(W.T @ W @ H)
    H = num / denom * H

    ## Normalize (W D^-1) (D H)
    W_norms = np.linalg.norm(W, axis=0)
    D = np.diag(W_norms)
    D_inv = np.diag(1./W_norms)
    # print(W.shape, H.shape, D.shape)
    # print(W)
    # print(H)
    # print(D)
    W = W @ D_inv
    H = D @ H


## Inference
community = []
for i in range(n):
    comm = np.argmax(H[:,i])
    community.append(comm)

for i in range(num_comms):
    print("Community {} ".format(i), np.arange(num_nodes)[np.array(community) == i])
