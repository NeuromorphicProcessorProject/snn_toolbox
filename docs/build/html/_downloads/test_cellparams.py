# -*- coding: utf-8 -*-
"""
Simple visualization of how the membrane potential of one IF neuron varies with
different cell parameters.

Created on Wed Mar  9 18:16:00 2016

@author: rbodo
"""

if __name__ == '__main__':

    import readline  # Bugfix when importing pyNN
    import pyNN.nest as sim
    import matplotlib.pyplot as plt
    import numpy as np

    def plot_spiketrains(segments):
        for (num, segment) in enumerate(segments):
            y = np.ones_like(segment.spiketrains[0]) * num
            plt.plot(segment.spiketrains[0], y, '.',
                     label=str(segment.annotations[par]))
            plt.xlim(0, simparams['duration'])
            plt.xlabel('time (ms)')
        plt.legend(loc="upper left")
        plt.ylim(-0.1, num + 0.1)

    cellparams = {'v_thresh': 1,
                  'v_reset': 0,
                  'v_rest': 0,
                  'e_rev_E': 10,
                  'e_rev_I': -10,
                  'i_offset': 0,
                  'cm': 0.09,
                  'tau_m': 1000,
                  'tau_refrac': 0,
                  'tau_syn_E': 0.01,
                  'tau_syn_I': 0.01}
    simparams = {'duration': 100,
                 'dt': 1,
                 'delay': 1,
                 'input_rate': 1000}

    sim.setup()

    inp = sim.Population(1, sim.SpikeSourcePoisson(
        duration=simparams['duration'], rate=simparams['input_rate']))

    inp.label = 'input cell'
    outp = sim.Population(1, sim.IF_cond_exp, cellparams=cellparams)
    outp.label = 'output cell'

    inp.record('spikes')
    outp.record(['v', 'spikes'])

    synapse = sim.StaticSynapse(weight=0.5, delay=simparams['delay'])

    connector = sim.OneToOneConnector()

    connection = sim.Projection(inp, outp, connector, synapse)

    par = 'i_offset'
    for p in [0, 0.1, 1]:
        outp.set(**{par: p})
        cellparams[par] = p
        inp.set(rate=simparams['input_rate'])
        outp.initialize(v=cellparams['v_rest'])
        sim.run(simparams['duration'])
        sim.reset(annotations={par: p})

    inp_data = inp.get_data()
    outp_data = outp.get_data()

    sim.end()

    plt.figure()
    plot_spiketrains(inp_data.segments)
    plt.show()

    plt.figure()
    plot_spiketrains(outp_data.segments)
    plt.show()

    plt.figure()
    for segment in outp_data.segments:
        vm = segment.analogsignalarrays[0]
        plt.plot(vm.times, vm,
                 label=str(segment.annotations[par]))
    plt.plot(vm.times, np.ones_like(vm.times) * cellparams['v_thresh'], 'r--',
             label='V_thresh')
    plt.plot(vm.times, np.ones_like(vm.times) * cellparams['v_reset'], 'b-.',
             label='V_reset')
    plt.ylim([cellparams['v_reset'] - 0.1, cellparams['v_thresh'] + 0.1])
    plt.legend(loc="upper left")
    plt.xlabel("Time (%s)" % vm.times.units._dimensionality)
    plt.ylabel("Membrane potential (%s)" % vm.units._dimensionality)
    plt.show()
