"""
Author: Maosen Li, Shanghai Jiao Tong University
"""

import tensorflow as  tf
import numpy as np

def unif_weight_init(shape, name=None):

    initial = tf.random_uniform(shape, minval=-np.sqrt(6.0/(shape[0]+shape[1])), maxval=np.sqrt(6.0/(shape[0]+shape[1])), dtype=tf.float32)

    return tf.Variable(initial, name=name)


def sample_gaussian(mean, diag_cov):

    z = mean+tf.random_normal(diag_cov.shape)*diag_cov

    return z


def sample_gaussian_np(mean, diag_cov):

    z = mean+np.random.normal(size=diag_cov.shape)*diag_cov
    
    return z


def gcn_layer_id(norm_adj_mat, W, b):

    return tf.nn.relu(tf.add(tf.sparse_tensor_dense_matmul(norm_adj_mat, W), b))


def gcn_layer(norm_adj_mat, h, W, b):

    return tf.add(tf.matmul(tf.sparse_tensor_dense_matmul(norm_adj_mat, h), W), b)


def sigmoid(x):

    return 1.0/(1.0+np.exp(-x))