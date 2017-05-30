
def preprocessing_function(x):
    import numpy as np
    from keras.applications.vgg16 import preprocess_input
    return preprocess_input(np.expand_dims(x, 0))
