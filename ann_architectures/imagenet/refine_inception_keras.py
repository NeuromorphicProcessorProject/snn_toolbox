"""Refine inception-v3."""

import os
import numpy as np
import json
from keras.applications.inception_v3 import InceptionV3
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from snntoolbox.io_utils.common import save_parameters

data_path = '/home/rbodo/.snntoolbox/Datasets/imagenet'
train_path = data_path + '/training'
test_path = data_path + '/validation'
save_path = data_path + '/GoogLeNet'
class_idx_path = save_path + '/imagenet_class_index.json'

# create the base pre-trained model
model = InceptionV3()

# we use SGD with a low learning rate
model.compile(SGD(lr=0.0001, momentum=0.9), 'categorical_crossentropy',
              metrics=['accuracy'])

# Training parameters
target_size = (299, 299)
batch_size = 20
samples_per_epoch = 10000
num_val_samples = 500
num_epochs = 50

print('Loading dataset...')

x_test = np.load(
    os.path.join(data_path, 'inception_keras', 'x_test.npz'))['arr_0']
y_test = np.load(
    os.path.join(data_path, 'inception_keras', 'y_test.npz'))['arr_0']

loss, acc = model.evaluate(x_test, y_test)

print(acc)

class_idx = json.load(open(class_idx_path, "r"))
classes = [class_idx[str(idx)][0] for idx in range(len(class_idx))]

datagen = ImageDataGenerator()
train_dataflow = datagen.flow_from_directory(train_path,
                                             target_size=target_size,
                                             classes=classes,
                                             batch_size=batch_size)

test_dataflow = datagen.flow_from_directory(test_path,
                                            target_size=target_size,
                                            classes=classes,
                                            batch_size=num_val_samples)

model.fit_generator(train_dataflow)

model.save('/home/rbodo/.snntoolbox/data/imagenet/inception/'
           'inception_refined.h5')
