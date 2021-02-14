"""
Usage:
For semisupervised attack:
    python -u simple_all.py cora semisup > all_logs/all_s_cora_log.txt
For supervised attack:
    python -u simple_all.py cora > all_logs/all_cora_log.txt
"""

import networkx as nx
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import copy
import sys
from sklearn import metrics
from node_embedding_attack.utils import load_dataset, standardize, generate_candidates_addition, generate_candidates_removal
from node_embedding_attack.perturbation_attack import baseline_random_top_flips, baseline_degree_top_flips, perturbation_top_flips
from node_embedding_attack.utils import flip_candidates, predict_L
from node_embedding_attack.embedding import deepwalk_skipgram, deepwalk_svd, node2vec_snap, line_cversion
from node_embedding_attack.utils import evaluate_embedding_node_classification, evaluate_embedding_link_prediction

def load_npz(fileloc, attack=None, ctype='addition'):
    G = load_dataset(fileloc)
    adj_matrix, labels = standardize(G['adj_matrix'], G['labels'])

    g = nx.Graph(adj_matrix)
    labels = labels
    feats = np.eye(len(g.nodes()))
    train_mask = np.zeros(len(g.nodes))

    sindices = np.random.choice(np.arange(0, len(train_mask), 1),
                                size=int(0.1 * len(train_mask)), replace=False)

    train_mask[sindices] = 1

    test_mask = np.ones(len(g.nodes))
    return adj_matrix, feats, labels, train_mask, test_mask


def get_candidates(adj_matrix, ctype='addition', numcds=5000):
    if ctype == 'addition':
        candidates = generate_candidates_addition(
            adj_matrix=adj_matrix, n_candidates=numcds)
    elif ctype == 'removal':
        candidates = generate_candidates_removal(adj_matrix=adj_matrix)
    elif ctype == 'combined':
        candidates1 = generate_candidates_addition(
            adj_matrix=adj_matrix, n_candidates=numcds)
        candidates2 = generate_candidates_removal(adj_matrix=adj_matrix)
        candidates = np.concatenate([candidates1, candidates2])
    return candidates


def get_attacked_graph(adj_matrix, candidates, attack=None, nflips=None, dim=None, window_size=None, L=None):
    if attack is not None:
        if attack == 'rnd':
            flips = baseline_random_top_flips(candidates, n_flips, 0)
        elif attack == 'deg':
            flips = baseline_degree_top_flips(
                adj_matrix, candidates, n_flips, True)
        elif attack == 'our':
            flips = perturbation_top_flips(
                adj_matrix, candidates, n_flips, dim, window_size, L)
        elif attack == 'ori':
            flips = perturbation_top_flips(
                adj_matrix, candidates, n_flips, dim, window_size, mode='unsup')
        adj_matrix_flipped = flip_candidates(adj_matrix, flips)
        return adj_matrix_flipped
    return adj_matrix


def get_embedding(method, adj_matrix, dim=None, window_size=None, ):
    if method == 'deepwalk_svd':
        embs, _, _, _ = deepwalk_svd(adj_matrix, window_size, dim)
        return embs
    elif method == 'deepwalk_skipgram':
        embs = deepwalk_skipgram(adj_matrix, embedding_dim=dim, window_size=window_size)
        return embs
    elif method == 'node2vec':
        embs = node2vec_snap(adj_matrix, embedding_dim=dim, window_size=window_size)
        return embs
    elif method == 'line':
        embs = line_cversion(adj_matrix, embedding_dim=dim)
        return embs
    return None

if sys.argv[1] == 'ffire':
    dname = "./data/forest_fire_smallcommunity.gml"
    g = nx.read_gml(dname)
    graph = copy.deepcopy(g)
    amat = nx.adj_matrix(graph)
    label_dict = nx.get_node_attributes(graph, 'cval')
    label = np.ones(len(label_dict.keys()))
    for k in label_dict:
        label[int(k)] = label_dict[k]
    amat, labels = standardize(amat, label)
    features = np.eye(len(g.nodes()))
    train_mask = np.zeros(len(g.nodes))

    sindices = np.random.choice(np.arange(0, len(train_mask), 1),
                                size=int(0.1 * len(train_mask)), replace=False)

    train_mask[sindices] = 1

    test_mask = np.ones(len(g.nodes))
elif sys.argv[1] == 'lfr':
    mu = 0.3
    n = 1000
    tau1 = 3
    tau2 = 2
    g = nx.algorithms.community.LFR_benchmark_graph(
        n, tau1, tau2, mu, average_degree=20, min_community=200, seed=10)
    communities = {frozenset(g.nodes[v]['community']) for v in g}
    for i, community in enumerate(communities):
        for node in community:
            g.node[node]['cval'] = i
    
    graph = copy.deepcopy(g)
    amat = nx.adj_matrix(graph)
    label_dict = nx.get_node_attributes(graph, 'cval')
    label = np.ones(len(label_dict.keys()))
    for k in label_dict:
        label[int(k)] = label_dict[k]
    amat, labels = standardize(amat, label)
    features = np.eye(len(g.nodes()))
    train_mask = np.zeros(len(g.nodes))

    sindices = np.random.choice(np.arange(0, len(train_mask), 1),
                                size=int(0.1 * len(train_mask)), replace=False)
    
    # train and test masks are only used for GCNs
    train_mask[sindices] = 1
    test_mask = np.ones(len(g.nodes))
else:
    dname = "./data/{}.npz".format(sys.argv[1])
    amat, features, labels, train_mask, test_mask = load_npz(dname)
    g = nx.Graph(amat)

n_flips = 500
dim = 32
window_size = 5
L = (labels == np.unique(labels)[:, None]).astype(int).T

use_semisup = False
if len(sys.argv) >= 3 and sys.argv[2] == 'semisup':
    print('Using semisupervised mode!')
    embedding0 = deepwalk_skipgram(amat, dim, window_size=window_size)
    L_atk = predict_L(embedding0, labels)
    L_atk_onehot = (L_atk == np.unique(labels)[:, None]).astype(int).T
    L = L_atk_onehot

if len(sys.argv) >= 4 and sys.argv[3] == 'lp':
    absent_set = generate_candidates_addition(
        adj_matrix=amat, n_candidates=10*int(0.1*len(g.edges())))
    present_set = generate_candidates_removal(adj_matrix=amat)
    ps_idx = np.random.choice(len(present_set), int(0.1*len(g.edges())))
    present_set = present_set[ps_idx]
    node_pairs = np.concatenate([absent_set, present_set])
    np.random.shuffle(node_pairs)

print('Num Nodes:', amat.shape[0])
print('Num Edges:', np.sum(amat))

outdict = {}
for emethod in ['deepwalk_svd', 'deepwalk_skipgram', 'node2vec', 'line']:
# for emethod in ['deepwalk_svd']:
    output_tuples = []
    for ctype in ['addition', 'removal', 'combined']:
    # for ctype in ['addition']:
        for attacktype in [None, 'rnd', 'our', 'ori']:
            candidates = get_candidates(amat, ctype=ctype)
            adj_matrix = get_attacked_graph(
                amat, candidates, attack=attacktype, nflips=n_flips, dim=dim, window_size=window_size, L=L)
            embs = get_embedding(emethod, adj_matrix, dim=dim, window_size=window_size)
            if len(sys.argv) >=4 and sys.argv[3] == 'lp':
                if ctype != 'addition':
                    continue
                roc_auc, ap_score = evaluate_embedding_link_prediction(
                    adj_matrix, node_pairs, embs)
                acc = ap_score
            else:
                f1_scores_mean, _ = evaluate_embedding_node_classification(embs, labels)
                acc = f1_scores_mean[0]
            output_tuples.append((ctype, attacktype, acc))
    outdict[emethod] = output_tuples

print(dname)
for emname, output_tuples in outdict.items():
    print('\n\n')
    print(emname)
    for i in output_tuples:
        print(i)
print('\n')
