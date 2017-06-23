# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 14:50:36 2015

@author: rbodo
"""


if __name__ == '__main__':
    from keras.datasets import mnist
    import matplotlib.pyplot as plt
    from random import randint
    import numpy as np
    from time import sleep
    # For compatibility with python2
    from builtins import range

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    def plot_prediction(X_test, y_test):
        ind = randint(0, len(X_test))
        x = X_test[ind]
        y = y_test[ind]

        plt.figure(figsize=(5, 7))
        plt.imshow(x)
        plt.title('Input Image')
        plt.xlabel('prediction: ' +
                   np.str(np.argmax(ann.predict(x.reshape(1, 784)))) + '\n' +
                   'ground truth: ' + np.str(y))
        plt.show()

    for i in range(10):
        plot_prediction(X_test, y_test)
        sleep(1)
