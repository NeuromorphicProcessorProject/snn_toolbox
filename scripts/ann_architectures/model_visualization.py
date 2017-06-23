# coding=utf-8

"""Keras Model Visualization.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

import os

from keras.models import model_from_json
from keras.utils.visualize_util import plot

import snntoolbox
from snntoolbox.simulation.backends.inisim_backend import custom_layers

data_path = snntoolbox.toolbox_root

model_name = "snn_99.16_INI"
model_name = os.path.join(data_path, model_name)
model_json = model_name + ".json"
model_struct = model_name + ".h5"
model_vis_img = model_name + ".vis.png"

json_file = open(model_json, 'r')
# model = model_from_json(json_file.read())
model = model_from_json(json_file.read(),
                        custom_objects=custom_layers)
model.load_weights(model_struct)

plot(model, to_file=model_vis_img, show_shapes=True, show_layer_names=True)
