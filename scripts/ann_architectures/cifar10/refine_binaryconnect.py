from __future__ import absolute_import
from __future__ import print_function

import os
import numpy as np
from keras.models import load_model, Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import adam
from keras.regularizers import l2
from snntoolbox.utils.utils import ClampedReLU

path = '/home/rbodo/.snntoolbox/data/cifar10/binaryconnect'
dataset_path = '/home/rbodo/.snntoolbox/Datasets/cifar10/pylearn2_gcn_whitened'
tensorboard_path = os.path.join(path, 'training')

batch_size = 64
nb_epoch = 50

# Experiment 1
# label = 'relu'
# nonlinearity = 'relu'
# bias_regularizer = [None] * 9
# model_init = load_model(os.path.join(path, '91.91_parsed.h5'), compile=False)

# Experiment 2
# label = 'relu_0.1_1'
# nonlinearity = ClampedReLU(0.1, 1.0)
# bias_regularizer = [None] * 9
# model_list = os.listdir(os.path.join(tensorboard_path, 'relu'))
# print("Initializing with model {}.".format(model_list[-2]))
# model_init = load_model(os.path.join(tensorboard_path, 'relu',
#                                      model_list[-2]), compile=False)

# # Experiment 3
# label = 'relu_0.1_1_bias_regularizer5'
# nonlinearity = ClampedReLU(0.1, 1.0)
# bias_regularizer = [l2(0.05), l2(0.9), l2(0.05), l2(0.5), l2(0.5), l2(0.5),
#                     l2(0.5), l2(0.01), l2(0.01)]
# model_list = os.listdir(os.path.join(tensorboard_path, 'relu_0.1_1'))
# print("Initializing with model {}.".format(model_list[-2]))
# model_init = load_model(os.path.join(
#     tensorboard_path, 'relu_0.1_1', model_list[-2]),
#     {'clamped_relu_0.1_1.0': nonlinearity}, False)

# Experiment 4
label = 'relu_0.1_1_bias_regularizer7'
nonlinearity = ClampedReLU(0.1, 1.0)
bias_regularizer = [l2(0.01), l2(2.0), l2(0.05), l2(3.0), l2(0.5), l2(1.0),
                    l2(1.0), l2(0.001), l2(0.01)]
model_list = os.listdir(os.path.join(tensorboard_path,
                                     'relu_0.1_1_bias_regularizer6'))
print("Initializing with model {}.".format(model_list[-2]))
model_init = load_model(os.path.join(
    tensorboard_path, 'relu_0.1_1_bias_regularizer6', model_list[-2]),
    {'clamped_relu_0.1_1.0': nonlinearity}, False)

model = Sequential()

model.add(Conv2D(128, (3, 3), padding='same', activation=nonlinearity,
                 input_shape=(3, 32, 32), bias_regularizer=bias_regularizer[0]))
model.add(Conv2D(128, (3, 3), padding='same', activation=nonlinearity,
                 bias_regularizer=bias_regularizer[1]))
model.add(MaxPooling2D())

model.add(Conv2D(256, (3, 3), padding='same', activation=nonlinearity,
                 bias_regularizer=bias_regularizer[2]))
model.add(Conv2D(256, (3, 3), padding='same', activation=nonlinearity,
                 bias_regularizer=bias_regularizer[3]))
model.add(MaxPooling2D())

model.add(Conv2D(512, (3, 3), padding='same', activation=nonlinearity,
                 bias_regularizer=bias_regularizer[4]))
model.add(Conv2D(512, (3, 3), padding='same', activation=nonlinearity,
                 bias_regularizer=bias_regularizer[5]))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(1024, activation=nonlinearity,
                bias_regularizer=bias_regularizer[6]))
model.add(Dense(1024, activation=nonlinearity,
                bias_regularizer=bias_regularizer[7]))
model.add(Dense(10, activation='softmax',
                bias_regularizer=bias_regularizer[8]))

model.set_weights(model_init.get_weights())

optimizer = adam(0.00001)
model.compile(optimizer, 'categorical_crossentropy', metrics=['accuracy'])

x_train = np.load(os.path.join(dataset_path, 'x_train.npz'))['arr_0']
y_train = np.load(os.path.join(dataset_path, 'y_train.npz'))['arr_0']
x_test = np.load(os.path.join(dataset_path, 'x_test.npz'))['arr_0']
y_test = np.load(os.path.join(dataset_path, 'y_test.npz'))['arr_0']

score = model.evaluate(x_test, y_test)
print(score)

traingen = ImageDataGenerator(horizontal_flip=True, rotation_range=20,
                              width_shift_range=0.2, height_shift_range=0.2)

trainflow = traingen.flow(x_train, y_train, batch_size)

out_path = os.path.join(tensorboard_path, label)
tensorboard = TensorBoard(out_path, 2)
checkpointer = ModelCheckpoint(os.path.join(
    out_path, '{epoch:02d}-{val_acc:.2f}.h5'), 'val_acc', 1, True)

model.fit_generator(trainflow, len(x_train) / batch_size, nb_epoch,
                    callbacks=[checkpointer, tensorboard],
                    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test)
print('Test score:', score[0])
print('Test accuracy: {:.2%}'.format(score[1]))

model.save(os.path.join(out_path, '{:2.2f}.h5'.format(score[1] * 100)))
