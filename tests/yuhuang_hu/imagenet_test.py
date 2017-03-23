"""Test ImageNet model.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

from __future__ import print_function
import os
import argparse
import numpy as np

from keras.models import model_from_json

home_path = os.environ.get("HOME", '~/')
config_path = os.path.join(home_path, ".snntoolbox")
data_dir = os.path.join(config_path, "datasets")


def model_evaluation(model_name, data_path, x_data_path, y_data_path):
    """Evaluate ImageNet model.

    Parameters
    ----------
    model_name : str
        the name of the model
    """
    model_json = os.path.join(config_path, model_name+".json")
    model_data = os.path.join(config_path, model_name+".h5")

    X_path = os.path.join(data_dir, data_path, x_data_path+".npz")
    Y_path = os.path.join(data_dir, data_path, y_data_path+".npz")

    print ("[MESSAGE] Loading data...")
    data = np.load(X_path)["arr_0"]
    label = np.load(Y_path)["arr_0"]
    print ("[MESSAGE] X data shape ", data.shape)
    print ("[MESSAGE] Y data shape ", label.shape)
    print ("[MESSAGE] Data loaded.")

    print ("[MESSAGE] Loading model...")
    json_file = open(model_json, 'r')
    model = model_from_json(json_file.read())
    model.load_weights(model_data)
    model.compile(loss='categorical_crossentropy', optimizer=None,
                  metrics=['accuracy'])
    print ("[MESSAGE] Model Loaded...")

    print ("[MESSAGE] Evaluating model...")
    score = model.evaluate(data, label, batch_size=100)
    print ("[MESSAGE] Model evaluated...")

    print('Test score:', score[0])
    print('Test accuracy:', score[1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Visualization \
                                     by Yuhuang Hu")
    parser.add_argument("-m", "--model-name", type=str,
                        default="imagenet",
                        help="The name of the model")
    parser.add_argument("-d", "--data-path", type=str,
                        default="imagenet",
                        help="The dataset set.")
    parser.add_argument("-xd", "--x-data-path", type=str,
                        default="X_norm",
                        help="The dataset set.")
    parser.add_argument("-yd", "--y-data-path", type=str,
                        default="Y_norm",
                        help="The dataset set.")
    args = parser.parse_args()
    model_evaluation(**vars(args))
