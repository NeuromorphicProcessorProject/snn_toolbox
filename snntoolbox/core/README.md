# SNN_utils

Core modules of the SNN toolbox.


### File overview

* convert_to_SNN.py - Module used to convert an analog to a spiking neural network.
    Returns the spiking model in the simulator-independent language pyNN.
* run_SNN.py - Simulates a spiking network with IF units and Poisson input in pyNN, using
    a Simulator like Brian, NEST, NEURON, etc.
* util_keras.py - A collection of helper functions, including spiking layer classes derived from
    Keras layers, which were used to implement our own IF spiking simulator.
    Not needed when converting and running the SNN in pyNN.
* normalization - Normalize the weights of a network. The weights of each layer are normalized with respect to the maximum
    activation.
