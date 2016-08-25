"""Polting everything for investigating ANN.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

from keras.models import model_from_json
from keras import backend as K
import os
from os.path import join
import matplotlib.pyplot as plt
import numpy as np

from snntoolbox.io_utils.plotting import plot_layer_activity
from snntoolbox.core.inisim import custom_layers
np.set_printoptions(threshold=np.inf)

cifar10_label = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}


def plot_weights(model, filename, num_cols=8, num_rows=4):
    """Plot weights."""
    W = model.layers[0].W.get_value(borrow=True)

    plt.figure()
    for i in xrange(W.shape[0]):
        W_t = W[i]-np.min(W[i])
        W_t /= np.max(W_t)
        plt.subplot(num_cols, num_rows, i+1)
        plt.imshow(W_t.transpose(1, 2, 0), interpolation='nearest')
        plt.axis('off')

    plt.savefig(filename, bbox_inches='tight')

    print ("[MESSAGE] The filters are saved at %s" % (filename))


def plot_out(model, image, path, filename, layer=0):
    """plot activation from some layer."""
    get_act = K.function([model.layers[0].input, K.learning_phase()],
                         [model.layers[layer].output])
    layer_output = get_act([image, 0])[0]

    plot_layer_activity((layer_output[0], "1st convolution after relu"),
                        filename, path=path, limits=None)

    print ("[MESSAGE] The feature maps saved at %s"
           % (join(record_path, "fms_for_image_"+str(image_id)+".png")))


# data path
home_path = os.environ["HOME"]
config_path = join(home_path, ".snntoolbox")
data_path = join(config_path, "datasets")
cifar10_path = join(data_path, "cifar10")

# model path
model_name = "82.65.bodo"
model_json = join(config_path, model_name+".json")
model_data = join(config_path, model_name+".h5")

# output paths
out_path = join(config_path, "ann_investigate", model_name)
image_id = 300
record_path = join(out_path, "sample_image_"+str(image_id))
if not os.path.isdir(record_path):
    os.makedirs(record_path)
image_path = join(record_path, "sample_image_"+str(image_id)+".png")
filter_path = join(out_path, "filters.png")
fms_path = join(record_path, "fms_for_image_"+str(image_id)+".png")

# load data
data = np.load(os.path.join(cifar10_path, "X_test.npz"))["arr_0"]
label = np.load(os.path.join(cifar10_path, "Y_test.npz"))["arr_0"]

# plot image
image = np.array([data[image_id]])
plt.figure()
plt.imshow(image[0].transpose(1, 2, 0))
plt.title(cifar10_label[np.argmax(label[image_id])])
plt.savefig(image_path, bbox_inches='tight')
print ("[MESSAGE] sample image is saved at %s" % (image_path))

# load the model
json_file = open(model_json, 'r')
model = model_from_json(json_file.read(),
                        custom_objects=custom_layers)
model.load_weights(model_data)

# plot filters
if not os.path.exists(filter_path):
    plot_weights(model, filter_path, num_cols=4, num_rows=8)
plot_out(model, image, record_path,
         "fms_for_image_"+str(image_id)+".png", layer=1)
