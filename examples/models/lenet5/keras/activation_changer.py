import keras
from keras.layers import Activation
from keras import activations
from keras.models import load_model
from keras.datasets import mnist
import keras.utils as utils
import numpy as np
from keras.utils import np_utils

class NoisySoftplus():
    ''' The Noisy Softplus activation function
        Values of k and sigma taken from Liu et al. 2017
    
    '''
    
    def __init__(self, k=0.17, sigma=1):
        self.k = k
        self.sigma = sigma
        self.__name__ = 'noisy_softplus_{}_{}'.format(self.k,
                                                    self.sigma)
                
    def __call__ (self, *args, **kwargs):
        return self.k*self.sigma*keras.backend.softplus(args[0]/(self.k*self.sigma))


#loading model; stick h5 here.
filename = "98.96.h5"

model = load_model(filename)

model.summary()
#generating MNIST testing and training data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(Y_train, 10)
Y_test = np_utils.to_categorical(Y_test, 10)

X_train = np.moveaxis(X_train, 3, 1)
X_test = np.moveaxis(X_test, 3, 1)

#Evaluating pretrained model
score = model.evaluate(X_test, Y_test, verbose=0)
print('Original Test Loss:', score[0])
print('Original Test Accuracy:', score[1])


#Tinkering with model
new_model = type(model)()

new_model.inputs = model.inputs

for layer in model.layers:
    
    #changing activation
    if hasattr(layer, 'activation') and layer.activation == activations.relu:
        layer.activation = NoisySoftplus
             
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

model.save("modified_"+filename)
model = load_model("modified_"+filename, custom_objects ={'NoisySoftplus': NoisySoftplus()})
#model.summary()


'''
#Print activations (not shown in model.summary)
for layer in model.layers:
    if hasattr(layer, 'activation'):
        print(layer.activation)
'''

score = model.evaluate(X_test, Y_test, batch_size = 1, verbose=0)
print('NSP Test Loss:', score[0])
print('NSP Test Accuracy:', score[1])


epochs = 1

#Fine tuning training
print('Retraining network for %d epochs...' % epochs)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(X_train, Y_train,
                    batch_size=10,
                    epochs=epochs,
                    verbose=0,
validation_data=(X_test, Y_test))

print('Evaluating retrained network...')

score = model.evaluate(X_test, Y_test, batch_size = 100, verbose=0)
print('Retrained Test loss:', score[0])
print('Retrained Test accuracy:', score[1])

#model.save("model_retrained.h5")