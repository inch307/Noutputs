import argparse
import torch
import torch.nn as nn
import torch.optim as optim

import ml.linear as linear
import ml.logistc as logistic
import ml.svm as svm

import utils

parser = argparse.Argument()
parser.add_argument('--model', help='linear, logistic, svm')
parser.add_argument('--dataset', help='dataset')
# parser.add_argument('--input_size', help='input size of dataset')
# parser.add_argument('--output_size', help='output size of dataset')

parser.add_argument('--ldp', help='all, non, or name of mechanism', default='all')
parser.add_argument('--epsilon', type=float, help='privacy budget', default=4)
parser.add_argument('--discrt', action='store_true')
parser.set_defaults(distric=False)

parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--cv_n', type=int, default=5, help='n-times cross validation')
parser.add_argument('--cv_k', type=int, default=10, help='k-fold cross validation')
parser.add_argument('--lam', type=float, default=0.0001, help='regularization factor')
parser.add_argument('--epoch', type=int, default=3, help='local epoch')

# parser.add_argument('--rounds', type=int, default=30)
parser.add_argument('--total_nodes', type=int, default=10000)
parser.add_argument('--fraction', type=float, default=0.01, help='fraction of total nodes for each round')
# parser.add_argument('--batch_size', type=int, default=32, help='the batch size for each node')
# parser.add_argument('--num_users', type=int, default=32, help='num of users per epoch')


args = parser.parse_args()


# mean estimation dataset
# synthetic dataset with normal dist
# Pol, credit, adult

# linear regression dataset
# beijing housing**  https://www.kaggle.com/code/ericliu123/beijing-houseprice-predict   https://www.kaggle.com/datasets/ruiqurm/lianjia
# beijinj air qual**
# cal_housing
# synth

# https://arxiv.org/pdf/2208.07353.pdf
# WISDM Human Activity Recognition dataset 

# logistic and SVM dataset
# KDD99 or NSL-KDD** (binary)
# adult dataset, or credit card fraud detection**
# synth

X, y = utils.load_data(args)

# cross valid
# 


