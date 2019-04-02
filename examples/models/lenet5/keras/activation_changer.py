import keras
from keras import activations
from keras.models import load_model
from keras.utils.generic_utils import get_custom_objects
from keras.datasets import mnist
import keras.utils as utils
import numpy as np
from keras.utils import np_utils


def noisy_softplus(x, k=0.17, sigma=0.5):
    return sigma*k*keras.activations.softplus(x/(sigma*k))

get_custom_objects().update({'noisy_softplus': noisy_softplus})


#loading model
filename = "98.96.h5"

model = load_model(filename)

#model.summary()
#generating testing and training data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

X_train = np.moveaxis(X_train, 3, 1)
X_test = np.moveaxis(X_test, 3, 1)

#Evaluating pretrained model
score = model.evaluate(X_test, Y_test, verbose=0)
print('Original Test loss:', score[0])
print('Original Test accuracy:', score[1])


#Tinkering with model
new_model = type(model)()

new_model.inputs = model.inputs

for layer in model.layers:
    
    #changing activation
    if hasattr(layer, 'activation') and layer.activation == activations.relu:
        layer.activation = noisy_softplus
     
    '''#removing bias
    if hasattr(layer, 'use_bias') and layerconf['use_bias']:
        print("Found biased layer")
        layerconf['use_bias'] = False
        weights, biases = layer.get_weights()
        weights[weights<0.0] = 1.0
        
        weights = weights + np.abs(np.amin(weights))
        layerconf['weights'] = [weights]
        
    #enforcing non-negativity of weights
    if hasattr(layer, 'kernel_constraint'):
        layerconf['kernel_constraint'] = keras.constraints.NonNeg()
    '''
    '''
    #changing maxpooling to avgpooling
    if type(layer).__name__ == 'MaxPooling2D':
        layerconf['name'] = "avg_pooling2d_" + str(ap_counter)
        ap_counter += 1
        layer = keras.layers.AveragePooling2D.from_config(layerconf)
    else:
        layer = type(layer).from_config(layerconf)
    '''
#model = utils.apply_modifications(model)

model.save("modified"+filename)
model = load_model("modified"+filename, custom_objects={'noisy_softplus': noisy_softplus})
#model.summary()


'''
for layer in model.layers:
    if hasattr(layer, 'activation'):
        print(layer.activation)
'''
score = model.evaluate(X_test, Y_test, batch_size = 1, verbose=0)
print('NSP Test loss:', score[0])
print('NSP Accuracy:', score[1])