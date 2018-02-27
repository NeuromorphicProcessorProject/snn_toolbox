from __future__ import print_function

import os
import numpy as np
from pylearn2.utils import serial, string_utils
from pylearn2.datasets import preprocessing
from pylearn2.datasets.cifar10 import CIFAR10
from pylearn2.datasets.zca_dataset import ZCA_Dataset

path = '/mnt/2646BAF446BAC3B9/.snntoolbox/Datasets/cifar10'

data_dir = string_utils.preprocess(path)

print("Preparing output directory...")
output_dir = os.path.join(data_dir, 'pylearn2_gcn_whitened')
serial.mkdir(output_dir)

# print('Loading CIFAR-10 train dataset...')
# train_set = CIFAR10(which_set='train')
#
# print("Learning the preprocessor and preprocessing \
#       the unsupervised train data...")
# preprocessor = preprocessing.ZCA()
# train_set.apply_preprocessor(preprocessor=preprocessor, can_fit=True)
#
# print('Saving the unsupervised data')
# train_set.use_design_loc(output_dir+'/train.npy')
# serial.save(output_dir + '/train.pkl', train_set)
#
# print("Loading the test data")
# test_set = CIFAR10(which_set='test')
#
# print("Preprocessing the test data")
# test_set.apply_preprocessor(preprocessor=preprocessor, can_fit=False)
#
# print("Saving the test data")
# test_set.use_design_loc(output_dir+'/test.npy')
# serial.save(output_dir+'/test.pkl', test_set)

train_set = serial.load(os.path.join(output_dir, 'train.pkl'))
test_set = serial.load(os.path.join(output_dir, 'test.pkl'))

preprocessor = serial.load(os.path.join(output_dir, 'preprocessor.pkl'))

train_set = ZCA_Dataset(train_set, preprocessor, 0, 50000)
test_set = ZCA_Dataset(test_set, preprocessor)

train_set.X = train_set.X.reshape(-1, 3, 32, 32)
test_set.X = test_set.X.reshape(-1, 3, 32, 32)

# flatten targets
train_set.y = np.hstack(train_set.y)
test_set.y = np.hstack(test_set.y)

# Onehot the targets
train_set.y = np.float32(np.eye(10)[train_set.y])
test_set.y = np.float32(np.eye(10)[test_set.y])

np.savez_compressed(os.path.join(output_dir, 'x_train'), train_set.X)
np.savez_compressed(os.path.join(output_dir, 'y_train'), train_set.y)
np.savez_compressed(os.path.join(output_dir, 'x_test'), test_set.X)
np.savez_compressed(os.path.join(output_dir, 'y_test'), test_set.y)
