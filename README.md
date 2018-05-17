# Variational-Graph-Auto-Encoders
This is the implementation of paper "Variational Graph Auto-Encoder", which is published in NIPS 2016 Workshop.

Thomas N. Kipf, Max Welling, Variational Graph Auto-Encoders, In NIPS Workshop on Bayesian Deep Learning, 2016.

# How to run the code?
Insure that you have 4 GB memory in your GPU and you have installed the required module.

We have provided the training data in right form in './data', and the main program can be run directly, like input 'CUDA_VISIBLE_DEVICES=0 python main.py' in terminal. For the first time of running, the program will load the data and generate a graph, which might cost much time. After that, a loaded graph would be saved in a numpy file in './data', and you can load the graph more efficiently.

During the training process, the program would print the loss value and validation accuracy. At last, the ROC curve would be saved in './result'.

# Result
The result of our program is shown here.
![ROC_curve_citation](https://github.com/limaosen0/Variational-Graph-Auto-Encoders/blob/master/result/ROC_curve_citation.png)
![ROC_curve_facebook](https://github.com/limaosen0/Variational-Graph-Auto-Encoders/blob/master/result/ROC_curve_facebook.png)

# Tools and Modules
Tensorflow 1.5

igraph-python

numpy, scipy.sparse and matplotlib

# Notes

You can clone our program, but if you use it in your published paper, please cite the paper "Variational Graph Auto-Encoder".
