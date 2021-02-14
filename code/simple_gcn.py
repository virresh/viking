"""
Usage:
For semisupervised attack, node classification:
    python -u simple_gcn.py cora semisup > gcn_logs/gcn_s_cora_log.txt
For supervised attack, node classification:
    python -u simple_gcn.py cora normal
For semisupervised attack, link prediction:
    python -u simple_gcn.py cora semisup lp > gcn_logs/gcn_s_lp_cora_log.txt
For supervised attack, link prediction:
    python -u simple_gcn.py cora normal lp > gcn_logs/gcn_lp_cora_log.txt
"""

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from node_embedding_attack.embedding import deepwalk_skipgram
from node_embedding_attack.utils import evaluate_embedding_link_prediction

def gcn_msg(edges):
    # just pass source node's embedding to destination node
    return {'m': edges.src['h']}


def gcn_reduce(nodes):
    # sum the embedding of all neighbor nodes
    return {'h': torch.sum(nodes.mailbox['m'], dim=1)}


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        # define a fully connected layer to store W
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, feature):
        # perform one-pass of updates on graph
        # return the updated embeddings of all nodes
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata['h']
            return self.linear(h)


class Net(nn.Module):
    # Two layer GCN for prediction on 34-feature network
    # prediction two classes
    def __init__(self, insz, outsz):
        super(Net, self).__init__()
        self.layer1 = GCNLayer(insz, 128)
        self.layer2 = GCNLayer(128, outsz)

    def forward(self, g, features):
        # input features are being used to learn some vector per node
        x = F.relu(self.layer1(g, features))
        # learnt vector is refined by non-linear activation and used
        # to learn next a vector on next layer
        x = self.layer2(g, x)
        return x
    
    def get_embeddings(self, g, features):
        x = self.layer1(g, features)
        return x


def evaluate(model, g, features, labels, mask):
    model.eval()
    # disable gradient computation
    with torch.no_grad():
        # compute embeddings
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        # predict class 1 for node x if logits[x][1] > logits[x][0]
        _, indices = torch.max(logits, dim=1)
        ans = metrics.f1_score(labels, indices, average='micro')
        # correct = torch.sum(indices == labels)
        # # accuracy computation
        # return correct.item() * 1.0 / len(labels)
        return ans

def evaluate_lp(model, g, features, node_pairs, adj_matrix):
    model.eval()
    with torch.no_grad():
        embeddings = model.get_embeddings(g, features)
        roc, auc = evaluate_embedding_link_prediction(adj_matrix, node_pairs, embeddings)
        return auc

def load_karate_club():
    # generate training and testing masks
    # alongisde loading dataset
    g = nx.karate_club_graph()
    labels = []
    for i in g.nodes():
        n = g.node.data()[i]
        if n['club'] == 'Officer':
            labels.append(1)
        else:
            labels.append(0)

    # one-hot encoded node id
    feats = np.eye(len(g.nodes()))
    train_mask = np.zeros(len(g.nodes))
    # only first and last node for training
    train_mask[[0, train_mask.shape[0]-1]] = 1
    # all nodes for testing
    test_mask = np.ones(len(g.nodes))

    # convert everything to pytorch variables
    g = dgl.DGLGraph(g)
    feats = torch.FloatTensor(feats)
    train_mask = torch.BoolTensor(train_mask)
    test_mask = torch.BoolTensor(test_mask)
    labels = torch.LongTensor(labels)
    return g, feats, labels, train_mask, test_mask


def load_npz(fileloc, attack=None, ctype='addition'):
    G = load_dataset(fileloc)
    adj_matrix, labels = standardize(G['adj_matrix'], G['labels'])
    adj_matrix[np.nonzero(adj_matrix)] = 1
    
    g = dgl.DGLGraph(adj_matrix)
    labels = labels
    feats = np.eye(len(g.nodes()))
    train_mask = np.zeros(len(g.nodes))

    sindices = np.random.choice(np.arange(0, len(train_mask), 1),
                     size=int(0.1 * len(train_mask)), replace=False)

    train_mask[sindices] = 1

    test_mask = np.ones(len(g.nodes))

    feats = torch.FloatTensor(feats)
    train_mask = torch.BoolTensor(train_mask)
    test_mask = torch.BoolTensor(test_mask)
    labels = torch.LongTensor(labels)
    return adj_matrix, feats, labels, train_mask, test_mask


def get_candidates(adj_matrix, ctype='addition', numcds=5000):
    if ctype == 'addition':
        candidates = generate_candidates_addition(adj_matrix=adj_matrix, n_candidates=numcds)
    elif ctype == 'removal':
        candidates = generate_candidates_removal(adj_matrix=adj_matrix)
    elif ctype == 'combined':
        candidates1 = generate_candidates_addition(adj_matrix=adj_matrix, n_candidates=numcds)
        candidates2 = generate_candidates_removal(adj_matrix=adj_matrix)
        candidates = np.concatenate([candidates1, candidates2])
    return candidates


def get_attacked_graph(adj_matrix, candidates, attack=None, nflips=None, dim=None, window_size=None, L=None):
    if attack is not None:
        if attack == 'rnd':
            flips = baseline_random_top_flips(candidates, n_flips, 0)
        elif attack == 'deg':
            flips = baseline_degree_top_flips(adj_matrix, candidates, n_flips, True)
        elif attack == 'our':
            flips = perturbation_top_flips(adj_matrix, candidates, n_flips, dim, window_size, L)
        elif attack == 'ori':
            flips = perturbation_top_flips(adj_matrix, candidates, n_flips, dim, window_size, mode='unsup')
        adj_matrix_flipped = flip_candidates(adj_matrix, flips)
        return adj_matrix_flipped
    return adj_matrix


# g, features, labels, train_mask, test_mask = load_karate_club()

# dname = os.environ.get('GCNNPZ', 
#                        '/media/Common/ResearchWork/NetworkEmbeddings/saanp-withgcn/data/citeseer.npz')
                    #    '/media/Common/ResearchWork/NetworkEmbeddings/node_embedding_attack/data/cora.npz')


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
    train_mask = np.zeros(len(g.nodes))
    sindices = np.random.choice(np.arange(0, len(train_mask), 1),
                                size=int(0.1 * len(train_mask)), replace=False)
    train_mask[sindices] = 1
    
    features = torch.FloatTensor(np.eye(len(g.nodes())))
    train_mask = torch.BoolTensor(train_mask)
    test_mask = torch.BoolTensor(np.ones(len(g.nodes)))
    labels = torch.LongTensor(labels)
elif sys.argv[1] == 'lfr':
    dname = 'lfr'
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
    train_mask = np.zeros(len(g.nodes))
    sindices = np.random.choice(np.arange(0, len(train_mask), 1),
                                size=int(0.1 * len(train_mask)), replace=False)
    # train and test masks are only used for GCNs
    train_mask[sindices] = 1

    features = torch.FloatTensor(np.eye(len(g.nodes())))
    train_mask = torch.BoolTensor(train_mask)
    test_mask = torch.BoolTensor(np.ones(len(g.nodes)))
    labels = torch.LongTensor(labels)
else:
    dname = "./data/{}.npz".format(sys.argv[1])
    amat, features, labels, train_mask, test_mask = load_npz(dname)
    g = nx.Graph(amat)

# n_flips = 500
# dim = 32
# window_size = 5
# L = (labels == np.unique(labels)[:, None]).astype(int).T
# dname = "./data/{}.npz".format(sys.argv[1])
# amat, features, labels, train_mask, test_mask = load_npz(dname)

n_flips = 1000
dim = 32
window_size = 5
L = (labels.numpy() == np.unique(labels.numpy())[:, None]).astype(int).T

use_semisup = False
if len(sys.argv) >= 3 and sys.argv[2] == 'semisup':
    embedding0 = deepwalk_skipgram(amat, dim, window_size=window_size)
    L_atk = predict_L(embedding0, labels.numpy())
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


output_tuples = []

for ctype in ['addition', 'removal', 'combined']:
    if len(sys.argv) >= 4 and sys.argv[3] == 'lp' and ctype != 'removal':
        continue
    for attacktype in [None, 'rnd', 'our', 'ori']:
        candidates = get_candidates(amat, ctype=ctype)
        adj_matrix = get_attacked_graph(amat, candidates, attack=attacktype, nflips=1000, dim=dim, window_size=window_size, L=L)
        # print(net)

        # simple Adam optimizer. LR = 1e-2 because features are already in a small
        # range of 0 to 1
        g = dgl.DGLGraph(adj_matrix)
        net = Net(len(g.nodes), len(np.unique(labels)))
        optimizer = torch.optim.Adam(net.parameters(), lr=0.05)
        dur = []
        selected_epochs = set(range(0, 50, 5))
        to_visualize = []
        for epoch in range(25):
            if epoch >= 3:
                t0 = time.time()

            net.train()
            logits = net(g, features)
            logp = F.log_softmax(logits, 1)
            loss = F.nll_loss(logp[train_mask], labels[train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch >= 3:
                dur.append(time.time() - t0)

            acc = evaluate(net, g, features, labels, test_mask)

            if epoch in selected_epochs:
                to_visualize.append((epoch, logits.detach().numpy(), acc))
            print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(
                epoch, loss.item(), acc, np.mean(dur)))
        if len(sys.argv) >= 4 and sys.argv[3] == 'lp':
            acc = evaluate_lp(net, g, features, node_pairs, adj_matrix)
        output_tuples.append((ctype, attacktype, acc))
print("\n\n\n")
print(dname)
for i in output_tuples:
    print(i)
