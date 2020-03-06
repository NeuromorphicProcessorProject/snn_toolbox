import sys
import keras
print(keras.__version__)
import tensorflow
from snntoolbox.bin.utils import update_setup, run_pipeline

print(tensorflow.__version__)

sys.path.append('/homes/rbodo/Repositories/nxtf/nxsdk-apps')
filepath = '/homes/rbodo/Repositories/snn_toolbox_loihi/realtaste/keras/single_head/config'
config = update_setup(filepath)
run_pipeline(config)
