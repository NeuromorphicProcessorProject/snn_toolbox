import keras
import numpy as np
import os.path
from distutils.dir_util import copy_tree

current_path = os.path.abspath(os.path.dirname(__file__))
source_path = os.path.join(current_path, "../../datasets/mnist/")
destination_path = os.path.join(current_path, "dataset/")
print(source_path)
print(destination_path)
copy_tree(source_path, destination_path)

x_test_data_path = os.path.join(destination_path, "x_test.npz")
x_norm_data_path = os.path.join(destination_path, "x_norm.npz")

with np.load(x_test_data_path) as data:
    x_test = data['arr_0']

x_test = np.moveaxis(x_test, 1, 3)
with np.load(x_norm_data_path) as data:
    x_norm = data['arr_0']

x_norm = np.moveaxis(x_norm, 1, 3)

np.savez(x_test_data_path, x_test)
np.savez(x_norm_data_path, x_norm)

