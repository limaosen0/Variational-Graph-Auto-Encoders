"""
Author: Maosen Li, Shanghai Jiao Tong University
"""

import argparse
import os
import tensorflow as tf
import numpy as np

from model import VGAE
import File_Reader


parser = argparse.ArgumentParser(description='')

parser.add_argument('--dataset_dir', dest='dataset_dir', default='./data', help='path of the dataset')
parser.add_argument('--dataset_name', dest='dataset_name', default='citation', help='name of the dataset')
parser.add_argument('--result_dir', dest='result_dir', default='./result', help='result of the model testing')

parser.add_argument('--n_hidden', dest='n_hidden', type=int, default=200, help='dimension of hidden layer')
parser.add_argument('--dropout', dest='dropout', type=bool, default=True, help='Using dropout in training')
parser.add_argument('--keep_prob', dest='keep_prob', type=float, default=0.5, help='prob of keeping activitation nodes')
parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.05, help='initial learning rate')
parser.add_argument('--max_iteration', dest='max_iteration', type=int, default=1000, help='max iteration step')

args = parser.parse_args()

def main(_):

	if not os.path.exists(args.result_dir):
		os.makedirs(args.result_dir)

	if not os.path.exists(os.path.join(args.dataset_dir, 'sparse_graph_'+args.dataset_name+'.npz')):
		print("There is no graph stored. We load it now!!!")
		adjacency_mat, list_adjacency, _ = File_Reader.get_cluster(os.path.join(args.dataset_dir, args.dataset_name+'.txt'))
		File_Reader.save_sparse_csr(os.path.join(args.dataset_dir, 'sparse_graph_'+args.dataset_name+'.npz'), adjacency_mat)
	else:
		print("The graph has been stored!!!")
		adjacency_mat = File_Reader.load_sparse_csr(os.path.join(args.dataset_dir, 'sparse_graph_'+args.dataset_name+'.npz'))

	n_nodes = adjacency_mat.shape[0]

	config = tf.ConfigProto(allow_soft_placement=True)
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		model = VGAE(sess, n_nodes, args)
		model.train(args, adjacency_mat)


if __name__ == '__main__':
	tf.app.run()