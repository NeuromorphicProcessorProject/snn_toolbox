import pyNN.spiNNaker as sim
import numpy as np
import matplotlib as plt

sim.setup(timestep = 1)

parameters ={'tau_syn_I': 0.01, 'tau_refrac': 0.001, 'cm': 1.0, 'tau_syn_E': 0.01, 'tau_m': 1000.0, 'v_rest': 0.0, 'v_thresh': 0.01, 'v_reset': 0.0}

i_offsets = np.arange(0, 100, 1)
simtime = 10000
rates = []
pop = sim.Population(100, sim.IF_curr_exp, parameters)
pop.set(i_offset=i_offsets)

pop.record('spikes')

sim.run(simtime)

data = pop.get_data()

for spiketrain in data.segments[0].spiketrains:
    rate= len(spiketrain)/simtime/1000
    rates.append(rate)
    print(rate)

plt.figure()
plt.plot(i_offsets, rates)
plt.show()