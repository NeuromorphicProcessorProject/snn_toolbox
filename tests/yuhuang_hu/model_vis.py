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
    print ("===================================================")
    for layer in model.layers:
        if layer.__class__.__name__ == "Conv2D":
            print ("Layer Type: %s" % layer.__class__.__name__)
            print ("Filters shape: %d x %d x %d" %
                   (layer.filters, layer.kernel_size[1], layer.kernel_size[0]))
            print ("Stride: ", layer.strides)
            print ("Border mode: %s" % layer.padding)
        elif layer.__class__.__name__ == "Activation":
            print ("Activation type: %s" % layer.activation.__name__)
        elif layer.__class__.__name__ == "Dropout":
            print ("Dropout rate: %.2f" % layer.p)
        elif layer.__class__.__name__ == "Flatten":
            print ("Flatten layer")
        elif layer.__class__.__name__ == "Dense":
            print ("Layer Type: %s" % layer.__class__.__name__)
            print ("Output dimension: %d" % layer.units)
        elif layer.__class__.__name__ in ["MaxPooling2D", "AveragePooling2D"]:
            print ("Layer Type: %s" % layer.__class__.__name__)
            print ("Pooling size: ", layer.pool_size)
            print ("Stride: ", layer.strides)
            print ("Border mode: %s" % layer.padding)

        print ("---------------------------------------------------")
    print ("===================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Visualization \
                                     by Yuhuang Hu")
    parser.add_argument("-m", "--model-name", type=str,
                        default="99.16",
                        help="The name of the model")
    args = parser.parse_args()
    visualize_model(**vars(args))
