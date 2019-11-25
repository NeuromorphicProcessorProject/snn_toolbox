#Adapted from tutorial: https://elitedatascience.com/keras-tutorial-deep-learning-in-python

import numpy as np
from keras.models import Sequential
# importing layers permitted by the conversion tool
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, AveragePooling2D, DepthwiseConv2D, Flatten, Concatenate
from keras.utils import np_utils
from keras.datasets import mnist 
from matplotlib import pyplot as plt
#from keras import backend as K
#K.set_image_dim_ordering('tf')

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

     
model = Sequential()

#model.add(Conv2D(10, (5, 5), activation='relu', input_shape=(28,28,1)))
#print(model.output_shape)

model.add(Conv2D(16, (5,5), input_shape=(28,28,1), activation='relu', use_bias=True))
#model.add(DepthwiseConv2D((5,5), depth_multiplier = 4, activation='relu', use_bias=False))
#model.add(DepthwiseConv2D((5,5), depth_multiplier = 2, activation='relu', use_bias=True))
#model.add(DepthwiseConv2D((5,5), depth_multiplier = 2, activation='relu', use_bias=True))
#model.add(DepthwiseConv2D((5,5), depth_multiplier = 2, activation='relu', use_bias=True))

#model.add(MaxPooling2D(10))
model.add(Flatten())
model.add(Dense(384, activation='relu', use_bias=True))
#model.add(Flatten())
model.add(Dense(10, activation='softmax', use_bias=True))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=32, nb_epoch=1, verbose=1)
'''
filter = model.get_weights()[0]
filter = np.reshape(filter, (5,5))

plt.imshow(filter, cmap='gray_r')
plt.colorbar()
plt.show()'''
score = model.evaluate(X_test, Y_test, verbose=1)

print(score)

with open("PB_bias_test_2.json", "w") as text_file:
    text_file.write(model.to_json())
model.save_weights("PB_bias_test_2.h5")
