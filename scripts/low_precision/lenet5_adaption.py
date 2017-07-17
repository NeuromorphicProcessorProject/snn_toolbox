"""LeNet-5 with low precision activations

We use Moritz Milde's ADaPTION toolbox.

https://github.com/NeuromorphicProcessorProject/ADaPTION/blob/master/examples/
low_precision/quantization/network_quantization_and_conversion.ipynb

Make sure Moritz' low-precision fork of caffe is on PYTHONPATH when importing
caffe below.
"""


from caffe.quantization.net_descriptor import net_prototxt
from caffe.quantization.qmf_check import distribute_bits
from caffe.quantization.convert_weights import convert_weights

# initialize classes
d = distribute_bits()
n = net_prototxt()
c = convert_weights()

net_name = 'LeNet5'
n_bits_activations = 4
n_bits_weights = 16

caffe_root = '/mnt/2646BAF446BAC3B9/Repositories/caffe_lp/'
weight_dir = caffe_root + 'examples/low_precision/mnist/lenet5/'
model_dir = 'examples/low_precision/mnist/lenet5/'
script_dir = caffe_root + 'examples/low_precision/mnist/lenet5/'
layer_dir = 'examples/create_prototxt/layers/'
save_dir = caffe_root + 'examples/low_precision/mnist/lenet5/'

# Convert pre trained caffemodel to low precision blob architecture
c.convert_weights(net_name, caffe_root=caffe_root, weight_dir=weight_dir,
                  debug=True, model_dir=model_dir)

# We first have to estimate the bit distribution for weights and activations
# since the network will be constructed based on this estimate.
# The naming convention is:
#   NetworkName_deploy.prototxt (for test/one-time rounding)
#   NetworkName_train.prototxt (for finetune/re-train)
bit_w, net = d.weights(net_name=net_name, n_bits=n_bits_weights,
                       load_mode='high_precision', threshold=0.0,
                       caffe_root=caffe_root, model_dir=model_dir,
                       weight_dir=weight_dir, debug=True)
bit_a, net = d.activation(net_name=net_name, n_bits=n_bits_activations,
                          load_mode='high_precision', threshold=0.01,
                          caffe_root=caffe_root, model_dir=model_dir,
                          weight_dir=weight_dir, debug=True)

# Make sure the desired prototxt file to extract the net structure from is in
# the respective directory.
print("Extracting network structure")
net_layout = n.extract(net_name=net_name, mode='train', model=net,
                       caffe_root=caffe_root, weight_dir=weight_dir,
                       debug=True, model_dir=model_dir)

print("Creating new network based on weight/activation distribution")
n.create(net_name=net_name, net_descriptor=net_layout,
         bit_distribution_weights=bit_w, bit_distribution_act=bit_a, scale=True,
         init_method='xavier', lp=True, deploy=False, visualize=False,
         round_bias='false', rounding_scheme='STOCHASTIC',
         caffe_root=caffe_root, model_dir=model_dir, layer_dir=layer_dir,
         save_dir=save_dir, debug=True)
