import pyNN.spiNNaker as sim
import numpy as np
import matplotlib.pyplot as plt

sim.setup(timestep = 0.1)

parameters ={'tau_syn_I': 0.01,
              'tau_refrac': 0,
               'cm': 1.0,
                'tau_syn_E': 0.01,
                 'tau_m': 1000.0,
                  'v_rest': 0.0,
                  'v_thresh': 1.0,
                  'v_reset': 0.0}

i_offsets = np.arange(0, 0.1, 0.001)
simtime = 10000
rates = []
pop = sim.Population(100, sim.IF_curr_exp, parameters)
pop.set(i_offset=i_offsets)

pop.record('spikes')

sim.run(simtime)

data = pop.get_data()

for index, spiketrain in enumerate(data.segments[0].spiketrains):
    rate= len(spiketrain)/10.0
    rates.append(rate)
    print(rate)

v_thresh = pop.get('v_thresh')[0]

plt.figure()
plt.plot(i_offsets, rates)
plt.xlabel('i_offset/nA')
plt.ylabel('firing rate/Hz')
plt.title('v_thresh={:.2f}'.format(v_thresh))
plt.show()