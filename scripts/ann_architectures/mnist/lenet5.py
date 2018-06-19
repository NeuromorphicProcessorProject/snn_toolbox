# coding=utf-8

"""LeNet for MNIST"""

import os
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, TensorBoard

from snntoolbox.parsing.utils import \
    get_quantized_activation_function_from_string
from snntoolbox.utils.utils import ClampedReLU

batch_size = 32
epochs = 10

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32') / 255.
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32') / 255.
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

# nonlinearity = get_quantized_activation_function_from_string('relu_Q1.4')
# nonlinearity = ClampedReLU
nonlinearity = 'relu'

model = Sequential()

model.add(Conv2D(6, (5, 5), input_shape=(1, 28, 28), activation=nonlinearity))
model.add(MaxPooling2D())

model.add(Conv2D(16, (5, 5), activation=nonlinearity))
model.add(MaxPooling2D())
model.add(Dropout(0.5))

model.add(Conv2D(120, (5, 5), padding='same', activation=nonlinearity))

model.add(Flatten())
model.add(Dense(84, activation=nonlinearity))
model.add(Dense(10, activation='softmax'))

model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

path = '/home/rbodo/.snntoolbox/data/mnist/cnn/lenet5/keras/gradients'

checkpoint = ModelCheckpoint('weights.{epoch:02d}-{val_acc:.2f}.h5', 'val_acc')
gradients = TensorBoard(os.path.join(path, 'logs'), 2, write_grads=True)
callbacks = []  # [checkpoint, gradients]
model.fit(X_train, Y_train, batch_size, epochs,
          validation_data=(X_test, Y_test), callbacks=callbacks)

score = model.evaluate(X_test, Y_test)
print('Test score:', score[0])
print('Test accuracy:', score[1])

model.save(os.path.join(path, '{:2.2f}.h5'.format(score[1]*100)))
