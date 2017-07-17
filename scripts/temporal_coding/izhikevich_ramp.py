"""
Tests with the Izhikevich neuron model.

"""

import numpy as np
import matplotlib.pyplot as plt

import pyNN.nest as sim
from pyNN.utility.plotting import Figure, Panel

# === Configure the simulator ================================================

duration = 50
dt = 0.01

sim.setup(timestep=dt, min_delay=0.1)

# === Build and instrument the network =======================================

phasic_spiking = {'a': 0.02, 'b': 0.25, 'c': -65, 'd': 6}
class_2 = {'a': 0.2, 'b': 0.26, 'c': -65, 'd': 0}

params = phasic_spiking
n = 100
v_init = -64
increments = 1 * np.linspace(dt, 1 + dt, n)
ramps = []
for incr in increments:
    val = -0.5
    amplitudes = []
    for t in range(int(duration / dt)):
        amplitudes.append(val)
        val += incr
    ramps.append(amplitudes)

ramp_inputs = [sim.StepCurrentSource(times=np.arange(0, int(duration/dt)),
                                     amplitudes=ramp) for ramp in ramps]

neurons = sim.Population(n, sim.IF_cond_alpha(tau_m=1))  # sim.Izhikevich(**params))
for i, ramp_input in enumerate(ramp_inputs):
    ramp_input.inject_into(neurons[i:i+1])

neurons.record(['v', 'spikes'])
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

plt.scatter(increments, 1 / np.array(first_spiketimes), label='inverse ttfs')
# plt.scatter(increments, rates, label='avg spikerate')
plt.legend()
plt.savefig('FI')

v = data.filter(name="v")[0]

Figure(Panel(v, ylabel="Membrane potential (mV)", xticks=True,
             xlabel="Time (ms)", yticks=True))

# === Clean up and quit ========================================================

sim.end()
