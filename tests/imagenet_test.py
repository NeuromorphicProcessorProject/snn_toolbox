"""Test ImageNet model.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

from __future__ import print_function
import os
import argparse
import numpy as np

from keras.models import model_from_json

home_path = os.environ["HOME"]
config_path = os.path.join(home_path, ".snntoolbox")
data_dir = os.path.join(config_path, "datasets")


def model_evaluation(model_name, data_path):
    """Evaluate ImageNet model.

    Parameters
    ----------
    model_name : str
        the name of the model
    """
    model_json = os.path.join(config_path, model_name+".json")
    model_data = os.path.join(config_path, model_name+".h5")

    data_path = os.path.join(data_dir, data_path, "X_norm.npz")
    label_path = os.path.join(data_dir, data_path, "Y_norm.npz")

    print ("[MESSAGE] Loading data...")
    data = np.load(data_path)["arr_0"]
    label = np.load(label_path)["arr_0"]
    print ("[MESSAGE] Data loaded.")

    print ("[MESSAGE] Loading model...")
    json_file = open(model_json, 'r')
    model = model_from_json(json_file.read())
    model.load_weights(model_data)
    print ("[MESSAGE] Model Loaded...")

    print ("[MESSAGE] Evaluating model...")
    score = model.evaluate(data, label, batch_size=1000)
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
    args = parser.parse_args()
    model_evaluation(**vars(args))
