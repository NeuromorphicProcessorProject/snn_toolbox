# coding=utf-8

"""

AlexNet retrained with clamped relu and bias regularization.

"""

from __future__ import absolute_import
from __future__ import print_function

import os
import json
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import top_k_categorical_accuracy
from keras.callbacks import ModelCheckpoint, TensorBoard


num_epoch = 10
batch_size = 100
target_size = (224, 224)

data_path = '/home/rbodo/.snntoolbox/Datasets/imagenet'
train_path = os.path.join(data_path, 'training')
test_path = os.path.join(data_path, 'validation')
class_idx_path = os.path.join(data_path, 'imagenet_class_index_1000.json')
log_path = '/home/rbodo/.snntoolbox/data/imagenet/alexnet/ttfs/training'

# Experiment 1
label = 'base'
use_bias = True

model = Sequential()

model.add(Conv2D(96, (12, 12), strides=(4, 4), use_bias=use_bias,
                 activation='relu',
                 input_shape=(3, target_size[0], target_size[1])))
model.add(Conv2D(256, (5, 5), activation='relu', use_bias=use_bias))
model.add(MaxPooling2D())

model.add(Conv2D(384, (3, 3), activation='relu', use_bias=use_bias))
model.add(MaxPooling2D())

model.add(Conv2D(384, (3, 3), activation='relu', use_bias=use_bias))
model.add(Conv2D(256, (3, 3), activation='relu', use_bias=use_bias))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(4096, activation='relu', use_bias=use_bias))
model.add(Dense(4096, activation='relu', use_bias=use_bias))
model.add(Dense(1000, activation='softmax', use_bias=use_bias))

model.compile('adam', 'categorical_crossentropy',
              metrics=['accuracy', top_k_categorical_accuracy])

# Data set
num_train = 1281167
num_test = 50000
class_idx = json.load(open(class_idx_path, "r"))
classes = [class_idx[str(idx)][0] for idx in range(len(class_idx))]

traingen = ImageDataGenerator(rescale=1/255.)
trainflow = traingen.flow_from_directory(
    train_path, target_size, classes=classes, batch_size=batch_size)
testgen = ImageDataGenerator(rescale=1/255.)
testflow = testgen.flow_from_directory(
    test_path, target_size, classes=classes, batch_size=num_test)

out_path = os.path.join(log_path, label)
tensorboard = TensorBoard(out_path, 1)
checkpointer = ModelCheckpoint(os.path.join(
    out_path, '{epoch:02d}-{val_acc:.2f}.h5'), 'val_acc', 1, True)

model.fit_generator(trainflow, num_train/batch_size, num_epoch,
                    callbacks=[checkpointer, tensorboard],
                    validation_data=testflow,
                    validation_steps=num_test/batch_size,
                    workers=8, use_multiprocessing=True)

score = model.evaluate_generator(testflow, steps=num_test/batch_size)
print('Test score:', score[0])
print('Test accuracy: {:.2%}'.format(score[1]))

model.save(os.path.join(out_path, '{:2.2f}.h5'.format(score[1] * 100)))
