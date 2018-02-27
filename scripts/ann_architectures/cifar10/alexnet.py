# coding=utf-8

"""

Small CNN for Cifar10 classification, adapted from
https://github.com/akrizhevsky/cuda-convnet2/blob/master/layers/layers-cifar10-11pct.cfg

Get to 11% error, using methodology described here:
https://code.google.com/p/cuda-convnet/wiki/Methodology

"""

from __future__ import absolute_import
from __future__ import print_function

import os
from keras.datasets import cifar10
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.regularizers import l2
from keras.constraints import Constraint, non_neg
from keras.initializers import random_uniform
from snntoolbox.utils.utils import ClampedReLU
import keras.backend as k

path = '/home/rbodo/.snntoolbox/data/cifar10/ttfs/training'

batch_size = 64
nb_epoch = 100

# # Experiment 1
# label = 'base'
# nonlinearity = 'relu'
# center = True
# scale = True
# activity_regularizer = None
# kernel_regularizer = None
# bias_regularizer = None

# # Experiment 2
# label = 'bn'
# nonlinearity = 'relu'
# center = False
# scale = False
# activity_regularizer = None
# kernel_regularizer = None
# bias_regularizer = None

# Experiment 3
label = 'relu'
nonlinearity = 'relu'  # ClampedReLU(0.1, 1)


class NegMinMaxNorm(Constraint):
    """MinMaxNorm weight constraint.

    Constrains the weights incident to each hidden unit
    to have the norm between a lower bound and an upper bound.

    # Arguments
        min_value: the minimum norm for the incoming weights.
        max_value: the maximum norm for the incoming weights.
        rate: rate for enforcing the constraint: weights will be
            rescaled to yield
            `(1 - rate) * norm + rate * norm.clip(min_value, max_value)`.
            Effectively, this means that rate=1.0 stands for strict
            enforcement of the constraint, while rate<1.0 means that
            weights will be rescaled at each step to slowly move
            towards a value inside the desired interval.
        axis: integer, axis along which to calculate weight norms.
            For instance, in a `Dense` layer the weight matrix
            has shape `(input_dim, output_dim)`,
            set `axis` to `0` to constrain each weight vector
            of length `(input_dim,)`.
            In a `Conv2D` layer with `data_format="channels_last"`,
            the weight tensor has shape
            `(rows, cols, input_depth, output_depth)`,
            set `axis` to `[0, 1, 2]`
            to constrain the weights of each filter tensor of size
            `(rows, cols, input_depth)`.
    """

    def __init__(self, min_value=0.0, max_value=1.0, rate=1.0, axis=0):
        self.min_value = min_value
        self.max_value = max_value
        self.rate = rate
        self.axis = axis

    def __call__(self, w):
        # norms = k.sqrt(k.sum(k.square(w), axis=self.axis, keepdims=True))
        # desired = (self.rate * k.clip(w, self.min_value, self.max_value) +
        #            (1 - self.rate) * w)
        # w *= (desired / (k.epsilon() + norms))
        # return w
        return w
        #return k.clip(w, self.min_value, self.max_value)

    def get_config(self):
        return {'min_value': self.min_value,
                'max_value': self.max_value,
                'rate': self.rate,
                'axis': self.axis}


# # Experiment 4
# label = 'kernel_regularizer'
# nonlinearity = ClampedReLU(0.1, 1)
# center = True
# scale = True
# activity_regularizer = None
# kernel_regularizer = l2(0.0001)
# bias_regularizer = l2(0.000001)

model = Sequential()

model.add(Conv2D(128, (5, 5), bias_constraint=NegMinMaxNorm(-1, 0, 0.1),
                # kernel_constraint=non_neg(),
              #   bias_initializer=random_uniform(-0.1, 0),
              #   kernel_initializer=random_uniform(0, 0.1),
                 input_shape=(3, 32, 32)))
model.add(Activation(nonlinearity))
model.add(MaxPooling2D())

model.add(Conv2D(128, (3, 3), bias_constraint=NegMinMaxNorm(-1, 0, 0.1)))
                 #kernel_constraint=non_neg(),
                # bias_initializer=random_uniform(-0.1, 0),
               #  kernel_initializer=random_uniform(0, 0.1)))
model.add(Activation(nonlinearity))
model.add(MaxPooling2D())

model.add(Conv2D(256, (3, 3), bias_constraint=NegMinMaxNorm(-1, 0, 0.1)))
                 #kernel_constraint=non_neg(),
                 #bias_initializer=random_uniform(-0.1, 0),
                 #kernel_initializer=random_uniform(0, 0.1)))
model.add(Activation(nonlinearity))

model.add(Conv2D(512, (4, 4), bias_constraint=NegMinMaxNorm(-1, 0, 0.1)))
                 #kernel_constraint=non_neg(),
                 #bias_initializer=random_uniform(-0.1, 0),
                 #kernel_initializer=random_uniform(0, 0.1)))
model.add(Activation(nonlinearity))

model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

out_path = os.path.join(path, label)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

traingen = ImageDataGenerator(rescale=1./255., featurewise_center=True,
                              featurewise_std_normalization=True,
                              zca_whitening=False, horizontal_flip=True,
                              rotation_range=20, width_shift_range=0.2,
                              height_shift_range=0.2)

traingen.fit(x_train/255.)

trainflow = traingen.flow(x_train, y_train, batch_size)

testgen = ImageDataGenerator(rescale=1./255., featurewise_center=True,
                             featurewise_std_normalization=True,
                             zca_whitening=False)
testgen.fit(x_test/255.)

testflow = testgen.flow(x_test, y_test, len(x_test))

validation_data = testflow.next()

tensorboard = TensorBoard(out_path, 10)
checkpointer = ModelCheckpoint(os.path.join(
    out_path, '{epoch:02d}-{val_acc:.2f}.h5'), 'val_acc', 1, True)

model.fit_generator(trainflow, len(x_train) / batch_size, nb_epoch,
                    callbacks=[checkpointer, tensorboard],
                    validation_data=validation_data)

score = model.evaluate_generator(testflow, len(x_test) / batch_size)
print('Test score:', score[0])
print('Test accuracy:', score[1])

model.save(os.path.join(out_path, '{:2.2f}.h5'.format(score[1] * 100)))
