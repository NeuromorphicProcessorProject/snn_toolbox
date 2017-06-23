# coding=utf-8

"""LeNet for MNIST"""

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D, MaxPooling2D

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32') / 255
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

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
model.fit(X_train, Y_train, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test)
print('Test score:', score[0])
print('Test accuracy:', score[1])

model.save('{:2.2f}.h5'.format(score[1]*100))
