# -*- coding: utf-8 -*-
"""
Create a small network of IF neurons to test if the measured spikerates match
the rates predicted by our theory of ANN-SNN-conversion.

Created on Wed Jul 13 08:16:02 2016

@author: rbodo
"""

import numpy as np
import matplotlib.pyplot as plt


# Fully-connected layer
class Layer():
    def __init__(self, input_shape, output_shape, name, thr, W, b, reset,
                 V0=0):
        self.input_shape = input_shape  # Number of units in previous layer
        self.output_shape = output_shape  # Number of units in this layer
        self.name = name  # Name of layer
        self.thr = thr  # Threshold
        self.W_shape = (output_shape, input_shape)  # Shape of weight matrix
        self.W = W  # Weights
        self.b = b  # Bias
        self.reset = reset  # The type of reset mechanism: 'zero' or 'subtract'
        self.V0 = V0  # Resting potential
        self.V = np.ones(self.output_shape) * self.V0  # Initialize to resting
        self.V_array = np.zeros((self.output_shape, len(times)))
        self.spiketrain = np.zeros((self.output_shape, len(times)))
        self.rates = np.zeros((self.output_shape, len(times)))
        self.pred_rates = np.zeros((self.output_shape, len(times)))

    def weighted_sum(self, x):
        return np.dot(self.W, x) + self.b

    def update_neurons(self, x, t, t_ind):
        z = self.thr * self.weighted_sum(x)  # Input to layer
        self.V += z  # Integrate in membrane potential
        spikes = self.V >= self.thr  # Trigger spikes
        self.spiketrain[:, t_ind] = spikes * t  # Write out spike times
        # Compute rates
        self.rates[:, t_ind] = [len(s.nonzero()[0])/t for s in self.spiketrain]
        self.V_array[:, t_ind] = self.V  # Write out V for plotting
        if self.reset == 'zero':
            self.V[spikes] = 0  # Reset to zero after spike
        elif self.reset == 'subtract':
            self.V[spikes] -= self.thr  # Subtract threshold after spike
        # Compute the rates predicted by theory
        self.pred_rates[:, t_ind] = self.get_pred_rates(x, t, t_ind)
        return spikes

    def set_activations(self, x):
        """Compute the activations of the corresponding ANN layer."""
        self.a = [max([0, z]) for z in self.weighted_sum(x)]

    def get_pred_rates(self, x, t, t_ind):
        """Compute the rates predicted by theory."""
        z = self.thr * self.weighted_sum(x)  # Input to first layer
        # For a higher layer, the input consists of the weighted sum of the
        # previous layer's firing rates
        q = np.dot(self.W, layers[0].pred_rates[:, t_ind]) + self.b / dt
        pred_rates = []
        # Iterate over the cells in the layer
        for i in range(len(z)):
            if self.reset == 'zero':
                if z[i] <= 0:  # Neuron cannot spike
                    pred_rates.append(0)
                else:
                    epsilon = z[i] - self.thr / (self.get_n(z[i]) + 1)
                    pred_rates.append((z[i] - epsilon) * (1 / dt - (self.V[i] -
                                      self.V0) / (z[i] * t)) / self.thr)
            elif self.reset == 'subtract':
                if self.name == 'Input':
                    pred_rates.append((z[i] / dt - (self.V[i] - self.V0) / t) /
                                      self.thr)
                elif self.name == 'Output':
                    pred_rates.append(q[i] - (self.V[i] - self.V0) / (t *
                                      self.thr))
        return pred_rates

    def get_n(self, z):
        """
        Compute the integer number n such that n*z < thr <= (n+1)*z, i.e.
        n times the input brings the membrane potential just below threshold.
        """
        n = 1
        while n * z < self.thr:
            n += 1
        return n - 1

    def plot_V(self):
        """
        Plot the membrane potential of all cell in the layer, together with the
        threshold and dashed indicating spikes.
        """
        plt.figure()
        plt.plot(times, np.ones_like(times) * self.thr, '--',
                 label='threshold', color='black')
        plt.plot(times, np.zeros_like(times), color='black')
        for i in range(self.output_shape):
            p, = plt.plot(times, self.V_array[i], '.', label='V'+str(i),
                          color=colors[i])
            spiketimes = (self.spiketrain[i].nonzero()[0] + 1) * dt + i/5
            plt.plot(spiketimes, np.zeros_like(spiketimes), '|',
                     label='spikes'+str(i), color=colors[i], markersize=10,
                     markeredgewidth=2)
        plt.title('Membrane Potential of Layer {}'.format(self.name))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel('time')
        plt.ylabel('V')
        plt.xlim(xmin=0)
        plt.ylim(ymin=low_lim, ymax=np.max([self.thr]+self.V_array)+0.1)
        plt.show()

    def plot_r(self):
        """
        Plot the measured and predicted spikerates, as well as the highest rate
        supported by the simulator, and the targeted ANN activations.
        """
        plt.figure()
        plt.plot(times, np.ones_like(times) / dt, color='black',
                 label='max rate')
        for i in range(self.output_shape):
            plt.plot([t+i/10 for t in times], self.rates[i], '.',
                     label='r'+str(i), color=colors[i])
            plt.plot([t+i/10 for t in times], self.pred_rates[i], 'x',
                     label='r'+str(i)+'pred', color=colors[i])
            plt.plot(times, np.ones_like(times) * self.a[i] / dt,
                     label='a'+str(i), color=colors[i])
        plt.title('Spikerates and Activations in Layer {}'.format(self.name))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlim(xmin=0)
        plt.xlabel('time')
        plt.ylim(ymin=-0.1, ymax=max([np.max(self.rates+self.pred_rates),
                                      max(self.a), 1/dt])+0.1)

    def print_config(self):
        """Print layer parameters."""
        print("Configuration of layer {}:".format(self.name))
        print("W: {}".format(self.W))
        print("b: {}".format(self.b))


def init_params(in_size, out_size, low_lim, high_lim):
    """Initialize layer parameters randomly."""
    W = (high_lim - low_lim) * np.random.rand(out_size, in_size) + low_lim
    b = (high_lim - low_lim) * np.random.rand(out_size) + low_lim
    return W, b


if __name__ == "__main__":
    in_size1 = 2  # Size of the input to the network
    out_size1 = 2  # Size of the first layer ("Input")
    in_size2 = out_size1
    out_size2 = 3  # Size of the second layer ("Output")
    low_lim = -1  # Interval for parameter initialization
    high_lim = 1
    T = 50  # Duration of simulation
    dt = 1  # Time resolution
    thr = 1  # Spike threshold
    times = range(dt, T, dt)  # List of time steps
    x = np.random.random_sample(in_size1)  # Input sample
#    x = [0.60937405, 0.18707114]
#    x = [0, 0]
#    colors = [plt.cm.spectral(i*100) for i in range(out_size2)]
    colors = ['r', 'g', 'b']  # For plotting
    reset = 'subtract'  # reset mechanism 'zero' or 'subtract'

    # Randomly initialize parameters
    W1, b1 = init_params(in_size1, out_size1, low_lim, high_lim)
    W2, b2 = init_params(in_size2, out_size2, low_lim, high_lim)

    # Or set fix
#    W1 = [[1.66878322, 1.29022261], [-1.18895428, 0.85312339]]
#    b1 = [1.12679297, -0.03022793]
#    W2 = [[0.45963593, 0.16917084], [1.33712331, -0.07807339],
#          [-0.97600214, -0.94265142]]
#    b2 = [-1.98658604, 1.85651182, -1.04564916]

    # Create two layers
    layers = [Layer(in_size1, out_size1, 'Input', thr, W1, b1, reset),
              Layer(in_size2, out_size2, 'Output', thr, W2, b2, reset)]

    # Print out parameters and input sample
    [l.print_config() for l in layers]
    print("Input: {}".format(x))

    # Compute the ANN activations
    layers[0].set_activations(x)
    layers[1].set_activations(layers[0].a)

    # Start simulation
    t_ind = 0
    for t in times:
        spikes = layers[0].update_neurons(x, t, t_ind)
        out_spikes = layers[1].update_neurons(spikes, t, t_ind)
        t_ind += 1

    # Display results
    for l in layers:
        l.plot_V()
        l.plot_r()
