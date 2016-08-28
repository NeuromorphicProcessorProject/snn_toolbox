"""Related functions for visualize model.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

from __future__ import print_function
import os
import argparse

from keras.models import model_from_json

home_path = os.environ["HOME"]
config_path = os.path.join(home_path, ".snntoolbox")
pref_dir = os.path.join(config_path, "preferences")


def visualize_model(model_name):
    """Model Visualiation.

    Parameters
    ----------
    model_name : str
        the name of the model
    """
    model_json = os.path.join(config_path, model_name+".json")
    # model_data = os.path.joinjoin(config_path, model_name+".h5")

    json_file = open(model_json, 'r')
    model = model_from_json(json_file.read())
    model.summary()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Visualization \
                                     by Yuhuang Hu")
    parser.add_argument("-m", "--model-name", type=str,
                        default="99.16",
                        help="The name of the model")
    args = parser.parse_args()
    visualize_model(**vars(args))
