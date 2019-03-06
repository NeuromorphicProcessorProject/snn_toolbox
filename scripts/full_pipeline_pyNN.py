from snntoolbox.bin.utils import update_setup
from snntoolbox.bin.utils import test_full


filepath = '/mnt/2646BAF446BAC3B9/Data/snn_conversion/mnist/cnn/lenet5/keras' \
           '/pyNN/channels_first/log/gui/01/config'

config = update_setup(filepath)

test_full(config)
