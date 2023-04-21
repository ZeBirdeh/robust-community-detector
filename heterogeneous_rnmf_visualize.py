### Community Detection Algorithm
### Implementation based on "Towards Robust Community Detection via Extreme Adversarial Attacks"
### https://ieeexplore.ieee.org/abstract/document/9956362

import numpy as np
from numpy import linalg
from scipy import sparse
from scipy.sparse.linalg import svds
import networkx as nx
from matplotlib import pyplot as plt
import itertools

rng : np.random.Generator = np.random.default_rng(seed=4237502)

## Read input from file
DIRECTORY_PATH = "multiview_data_20130124/politicsie/"
ID_FILEPATH = "politicsie.ids"
LINK_FILEPATH = ["politicsie-follows.mtx", "politicsie-mentions.mtx", "politicsie-retweets.mtx"]
CONTENT_FILEPATH = ["politicsie-listmerged500.mtx", "politicsie-lists500.mtx", "politicsie-tweets500.mtx"]
LEBEL_FILEPATH = "politicsie.communities"
OUTPUT_FILEPATH = "politicsie_output_cache"
num_comms = 7

## Read input from homogeneous graph
# FILEPATH = "dolphins_social_network"
# DIRECTORY_PATH = "homogeneous_graphs/"
# ID_FILEPATH = "dolphins.ids"
# LINK_FILEPATH = ["dolphin_links.mtx"]
# CONTENT_FILEPATH = []
# OUTPUT_FILEPATH = "dolphin_output"

user_ids = []
_index = {}
A = []
X = []

with open(DIRECTORY_PATH + ID_FILEPATH) as file:
    for i, line in enumerate(file):
        user_id = int(line.split()[0])
        user_ids.append(user_id)
        _index[user_id] = i

edgelists = []
for path in LINK_FILEPATH:
    train_i = []
    train_j = []
    train_val = []
    edges = []
    with open(DIRECTORY_PATH + path, "r") as file:
        num_users, num_users, _ = [int(val) for val in next(file).split()]
        for line in file:
            i, j, k = line.split()
            train_i.append(_index[int(i)])
            train_j.append(_index[int(j)])
            train_val.append(float(k))
            edges.append((int(i), int(j)))
        A.append(sparse.coo_matrix((train_val + train_val, (train_i + train_j, train_j + train_i)), shape=(num_users, num_users)).tocsr())

    ## Used to compile a list of edges
    edgelists.append(edges)

for path in CONTENT_FILEPATH:
    train_i = []
    train_j = []
    train_val = []
    with open(DIRECTORY_PATH + path, "r") as file:
        num_features, num_users, _ = [int(val) for val in next(file).split()]
        # print(num_features)
        for line in file:
            i, j, k = line.split()
            train_i.append(_index[int(j)])
            train_j.append(int(i))
            train_val.append(float(k))
        X.append(sparse.coo_matrix((train_val, (train_i, train_j)), shape=(num_users, num_features)).tocsr())
    
label_comms = {}
with open(DIRECTORY_PATH + LEBEL_FILEPATH, "r") as file:
    for line in file:
        comm_name, users = line.split(" ")
        label_comms[comm_name] = [int(user) for user in users.split(",")]



## Display

community = [[] for _ in range(num_comms)]
with open(OUTPUT_FILEPATH) as file:
    for comm in range(num_comms):
        line = next(file)
        ids = line.split(": ")[1].split(", ")
        for id in ids:
            community[comm].append(int(id))

## Code for displaying homogeneous graphs
edgelist = [edge for graph in edgelists for edge in graph]

accuracy = 0
for perm in (itertools.permutations(range(num_comms))):
    num_correct = 0
    for i, comm in enumerate(label_comms):
        num_correct += len(set(community[perm[i]]) & set(label_comms[comm]))
    accuracy = max(accuracy, num_correct / num_users)
print("Model accuracy: {}".format(accuracy))

G = nx.Graph(edgelist)
# pos = nx.spring_layout(G, seed=3113794652)
colorlist = ["tab:red", "tab:blue", "tab:orange", "tab:green", "tab:pink", "tab:purple", "tab:gray", "tab:brown", "tab:olive", "tab:cyan"]
pos = {}
for i in range(num_comms):
    for node in community[i]:
        pos[node] = (100 * np.math.cos(2*i*np.math.pi / num_comms) + 20*rng.random(), 100 * np.math.sin(2*i*np.math.pi / num_comms) + 20*rng.random())
    nx.draw_networkx_nodes(G, pos, node_size=25, nodelist=community[i], node_color=colorlist[i])

nx.draw_networkx_edges(G, pos, width=0.2, alpha=0.2)
plt.tight_layout()
plt.axis("off")
plt.show()