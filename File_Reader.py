"""
Author: Maosen Li, Shanghai Jiao Tong University
"""

import numpy as np
import scipy.sparse as sp
import igraph as ig
import os


def get_nnodes(file_name):
    
    file=open(file_name, 'r')
    n_nodes=0
    for line in file:
        edge = line.split()
        edge = list(map(int, edge))
        if edge[0] > n_nodes:
            n_nodes = edge[0]
        if edge[1] > n_nodes:
            n_nodes = edge[1]        
    file.close()
    n_nodes = n_nodes+1
    return n_nodes


def load_graph(file_name):

    file = open(file_name, 'r')
    G = ig.Graph()
    n_nodes = get_nnodes(file_name)
    G.add_vertices(n_nodes)
    for line in file:
        edge = line.split()
        edge = list(map(int, edge))
        G.add_edge(edge[0], edge[1])
    file.close()
    return G


def get_cluster(file_name, idx=-2):

    file_name = os.getcwd()+file_name
    G = load_graph(file_name)
    G.vs["name"] = list(range(G.vcount())) # naming each node to be 0, 1, 2, ...
    if idx == -2:
        G = G.components().giant()
    if idx == -1:
        G = G.components().giant()
        G = G.community_multilevel().giant()       
    else:       
        com = G.community_multilevel()
        for i in range(com.__len__()) :
            if idx in com.subgraph(i).vs["name"]:
                G = com.subgraph(i)
                break
    edges = G.get_edgelist()
    n_nodes = G.vcount()
    row = []
    col = []
    data = []
    for edge in edges:
        row.extend([edge[0], edge[1]])
        col.extend([edge[1], edge[0]])
        data.extend([1, 1])
    adjacency = sp.coo_matrix((data, (row,col)), shape=(n_nodes,n_nodes))
    list_indices = G.vs["name"]
    return adjacency, edges, list_indices


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, row=array.row, col=array.col, shape=array.shape)
    pass


def load_sparse_csr(filename):
    loader = np.load(filename)
    return sp.csr_matrix((loader['data'], (loader['row'],loader['col'])), shape=loader['shape'])


def load_adjacency(file_name):

    file_name = os.getcwd()+file_name
    file = open(file_name, 'r')
    row = []
    col = []
    data = []
    n_nodes = 0
    for line in file:
        edge = line.split()
        edge = list(map(int, edge))
        if edge[0] > 15000 or edge[1] > 15000:
            continue
        if edge[0] > n_nodes:
            n_nodes = edge[0]
        if edge[1] > n_nodes:
            n_nodes = edge[1]        
        row.extend([edge[0], edge[1]])
        col.extend([edge[1], edge[0]])
        data.extend([1, 1])
    file.close()
    n_nodes = n_nodes+1
    adjacency = sp.coo_matrix((data, (row,col)), shape=(n_nodes,n_nodes))
    return adjacency


def normalize_adjacency(adjacency):

    coo_adjacency = sp.coo_matrix(adjacency)
    adjacency_ = coo_adjacency + sp.eye(coo_adjacency.shape[0])
    degree = np.array(adjacency_.sum(1))
    d_inv = sp.diags(np.power(degree, -0.5).flatten())
    normalized = adjacency_.dot(d_inv).transpose().dot(d_inv)
    
    return dense_to_sparse(normalized)


def dense_to_sparse(adjacency):

    coo_adjacency = sp.coo_matrix(adjacency)
    indices = np.vstack((coo_adjacency.row, coo_adjacency.col)).transpose()
    values = coo_adjacency.data
    shape = np.array(coo_adjacency.shape, dtype=np.int64)
    return indices, values, shape


def train_test_split(adjacency):

    n_nodes = adjacency.shape[0]
    coo_adjacency = sp.coo_matrix(adjacency)
    coo_adjacency_upper = sp.triu(coo_adjacency, k=1)
    sp_adjacency = dense_to_sparse(coo_adjacency_upper)
    edges = sp_adjacency[0]
    num_test = int(np.floor(edges.shape[0]/10.))
    num_val = int(np.floor(edges.shape[0]/10.))

    idx_all = list(range(edges.shape[0]))
    np.random.shuffle(idx_all)
    idx_test = idx_all[:num_test]
    idx_val = idx_all[num_test:(num_val + num_test)]

    test_edges_pos = edges[idx_test]
    val_edges_pos = edges[idx_val]
    train_edges = np.delete(edges, np.hstack([idx_test, idx_val]), axis=0)
    
    test_edges_neg = []
    val_edges_neg = []
    edge_to_add = [0, 0]
    
    while (len(test_edges_neg) < len(test_edges_pos)):
        n1 = np.random.randint(0, n_nodes)
        n2 = np.random.randint(0, n_nodes)
        if n1 == n2:
            continue        
        if n1 < n2:
            edge_to_add = [n1, n2]
        else:
            edge_to_add = [n2, n1]        
        if any((edges[:]==edge_to_add).all(1)):
            continue
        test_edges_neg.append(edge_to_add)
        
    while (len(val_edges_neg) < len(val_edges_pos)):
        n1 = np.random.randint(0, n_nodes)
        n2 = np.random.randint(0, n_nodes)
        if n1 == n2:
            continue        
        if n1 < n2:
            edge_to_add = [n1, n2]
        else:
            edge_to_add = [n2, n1]        
        if any((edges[:] == edge_to_add).all(1)):
            continue
        val_edges_neg.append(edge_to_add)
    row = []
    col = []
    data = []
    for edge in train_edges:
        row.extend([edge[0], edge[1]])
        col.extend([edge[1], edge[0]])
        data.extend([1, 1])
    train_adjacency = sp.coo_matrix((data, (row,col)), shape=(n_nodes,n_nodes))
    return train_adjacency, test_edges_pos, test_edges_neg, val_edges_pos, val_edges_neg