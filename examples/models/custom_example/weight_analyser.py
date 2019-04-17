import keras
import matplotlib.pyplot as plt
import numpy as np
import bitstring

'''Loading model from files'''

filename = "custom_example_test"

json_file = open((filename+".json"), 'r')
loaded_model_json = json_file.read()
json_file.close()


model = keras.models.model_from_json(loaded_model_json)

model.load_weights(filename+".h5")

'''Getting weights'''
'''
weights = model.layers[1].get_weights()[0]
print(weights.shape)
#weights = weights.flatten()
weights = weights[:,:]
print(weights.shape)

weights_sum = np.log(np.sum(weights,1))

pos_weights = np.abs(weights[weights<0])

norm_pos_weights = np.log(np.divide(pos_weights,np.min(pos_weights)))


neg_weights = np.abs(weights[weights>0])

norm_neg_weights = np.log(np.divide(neg_weights,np.min(neg_weights)))

#log_binweights = np.log10(abs(weights[weights>0]))'''
'''binweights = [bitstring.BitArray(float=num, length=32) for num in weights]

binweights_mantissae = [x.bin[1:24] for x in binweights] 
binweights_exponents = [x.bin[24:] for x in binweights]
#print(binweights_exponents)
exponents = [int(x, 2) for x in binweights_exponents]
mantissae = [int(x, 2) for x in binweights_mantissae]
#print exponents
'''
#print(log_binweights)

'''Getting intermediate layer activations'''

from keras.datasets import mnist
from keras.utils import np_utils
from keras import Model
 


(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 784)
X_test = X_test.reshape(X_test.shape[0], 784)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(index=0).output)

print(X_test[0].shape)
intermediate_output0 = intermediate_layer_model.predict(np.reshape(X_test[0],(1,784)))
intermediate_output1 = intermediate_layer_model.predict(np.reshape(X_test[1],(1,784)))
print(intermediate_output0/np.max(intermediate_output0))
intermediate_output_no_zero = np.reshape(intermediate_output0,(784))[np.nonzero(np.reshape(intermediate_output0,(784)))]
#print(intermediate_output_no_zero)

#plt.hist(norm_pos_weights, density =True, cumulative=False, bins=20, color = 'red', alpha=0.5)
#plt.hist(norm_neg_weights, density =True, cumulative=False, bins=20, color = 'blue', alpha=0.5)
#plt.hist(weights, density=True, cumulative=False, bins=128, alpha=0.9)
#plt.hist(weights_sum, density=True, cumulative=False, bins=128, alpha=0.9)
#plt.ylabel('Probability');
#plt.imshow(intermediate_output.reshape(28,28))
plt.hist(intermediate_output_no_zero, density=False, cumulative=False, bins=128, alpha=0.9)
plt.show()