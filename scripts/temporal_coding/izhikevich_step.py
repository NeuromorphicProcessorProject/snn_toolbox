"""
Tests with the Izhikevich neuron model.

"""

import numpy as np
import matplotlib.pyplot as plt

import pyNN.nest as sim
from pyNN.utility.plotting import Figure, Panel

# === Configure the simulator ================================================

duration = 100
dt = 0.01

sim.setup(timestep=dt, min_delay=0.1)

# === Build and instrument the network =======================================

phasic_spiking = {'a': 0.02, 'b': 0.25, 'c': -65, 'd': 6}
class_2 = {'a': 0.2, 'b': 0.26, 'c': -65, 'd': 0}

params = class_2
n = 100
v_init = -64

input_currents = 0.0005 * np.logspace(-4, 6, n, base=np.e)
neurons = sim.Population(n, sim.Izhikevich(i_offset=input_currents, **params))

neurons.record(['v', 'u', 'spikes'])
neurons.initialize(v=v_init, u=-params['b']*v_init)

# === Run the simulation =====================================================

sim.run(duration)

# === Save the results, optionally plot a figure =============================

data = neurons.get_data().segments[0]

first_spiketimes = []
rates = []
for spiketrain in data.spiketrains:
    if len(spiketrain) == 0:
        first_spiketimes.append(np.infty)
    else:
        first_spiketimes.append(spiketrain[0])
    rates.append(np.count_nonzero(spiketrain) / duration)

plt.scatter(input_currents, 1 / np.array(first_spiketimes),
            label='inverse ttfs')
plt.scatter(input_currents, rates, label='avg spikerate')
plt.legend()
plt.savefig('FI')

v = data.filter(name="v")[0]
u = data.filter(name="u")[0]
Figure(Panel(v, ylabel="Membrane potential (mV)", xticks=True,
             xlabel="Time (ms)", yticks=True),
       Panel(u, ylabel="u variable (units?)")).save('mem')

# === Clean up and quit ========================================================

sim.end()
