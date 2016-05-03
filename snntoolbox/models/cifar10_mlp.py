from __future__ import absolute_import
from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.constraints import maxnorm
import snntoolbox
from snntoolbox.io.plotting import plot_history
from snntoolbox.io.save import save_model
import numpy as np
import os

'''
    Train a simple deep NN on the cifar10 dataset.
'''

path = os.path.join(snntoolbox._dir, 'data', 'cifar10', 'mlp')

batch_size = 128
nb_classes = 10
nb_epoch = 50

data_augmentation = True

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
input_length = int(np.prod(X_train.shape[1:]))
X_train = X_train.reshape(X_train.shape[0], input_length)
X_test = X_test.reshape(X_test.shape[0], input_length)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Dense(512, input_shape=(input_length,), b_constraint=maxnorm(0)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(256, b_constraint=maxnorm(0)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10, b_constraint=maxnorm(0)))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

early_stopping = EarlyStopping(monitor='val_loss', patience=10)
checkpointer = ModelCheckpoint(filepath=path+'{epoch:02d}-{val_loss:.2f}.h5',
                               verbose=1, save_best_only=True)

history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                    show_accuracy=True, verbose=2,
                    validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, show_accuracy=True,
                       batch_size=batch_size)

print('Test score:', score[0])
print('Test accuracy:', score[1])
plot_history(history)

filename = '{:2.2f}'.format(score[1] * 100)
path = os.path.join(snntoolbox._dir, 'data', 'cifar10', 'mlp', filename)
save_model(model, path, 'ann_'+filename)
