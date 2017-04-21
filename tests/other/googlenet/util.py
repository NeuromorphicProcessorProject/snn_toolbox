import os
import numpy as np
import matplotlib.pyplot as plt


def get_input_voltage(spikes, weights, bias):
    return np.dot(spikes.transpose(), weights) + bias


def get_membrane_potential(input_t, reset, v_clip, t_clamp=0):
    v = 0
    v_t = [0] * t_clamp
    spikes_t = []
    for t in range(len(input_t)):
        if t < t_clamp:
            continue
        v += input_t[t]
        if v_clip:
            v = np.clip(v, -2, 2)
        if v >= 1:
            if reset == 'sub':
                v -= 1
            elif reset == 'zero':
                v = 0
            elif reset == 'mod':
                v %= 1
            spikes_t.append(t)
        v_t.append(v)
    return np.array(v_t), spikes_t


def plot_spiketrains(spiketimes_layers, labels, path):
    num_layers = len(spiketimes_layers)
    f, ax = plt.subplots(num_layers, 1, sharex=True, sharey=True)
    f.set_figwidth(10)
    f.set_figheight(10 * num_layers)
    ax[-1].set_xlabel('Simulation time')
    for i, spiketimes in enumerate(spiketimes_layers):
        ax[i].set_title(labels[i])
        ax[i].set_ylabel('Neuron index')
        for (neuron, spiketrain) in enumerate(spiketimes):
            spikelist = [j for j in spiketrain if j != 0]
            y = np.ones_like(spikelist) * neuron
            ax[i].plot(spikelist, y, '.')
    f.savefig(os.path.join(path, 'input_spiketrains'), bbox_inches='tight')


def get_rates(spiketimes, T, dt, t_clamp=0, idx=0):
    # ``idx`` is the feature index.
    rates = [0] * t_clamp
    count = 0
    t_idx = int(np.ceil(t_clamp/dt))
    for t in np.arange(t_clamp, T, dt):
        count += int(spiketimes[idx][t_idx] > 0)
        rates.append(count/(t+dt-t_clamp))
        t_idx += 1
    return rates


def plot_spikerates(spiketimes_layers, labels, path, T, dt, target_activations=None, t_clamp=None, idx=0, filename='spikerates'):
    plt.xlabel('Simulation time')
    plt.ylabel('Spike-rate')
    colors = ['blue', 'green', 'red']  # plt.cm.plasma(np.linspace(0, 0.9, len(labels)))
    if t_clamp is None:
        t_clamp = len(spiketimes_layers) * [0]
    for i, spiketimes in enumerate(spiketimes_layers):
        plt.plot(np.arange(0, T, dt), get_rates(spiketimes, t_clamp[i], idx), label=labels[i], color=colors[i])
        if target_activations:
            plt.plot([0, T], [target_activations[i], target_activations[i]], label='{}_target'.format(labels[i]), color=colors[i])
    plt.legend(loc='center right')
    plt.savefig(os.path.join(path, filename), bbox_inches='tight')


def plot_cumsum(input_t, labels, path, T, dt):
    plt.xlabel('Simulation time')
    plt.ylabel('Cumulated input to membrane potential')
    for i, inp in enumerate(input_t):
        plt.plot(np.arange(0, T, dt), np.cumsum(inp), label=labels[i])
    plt.legend()
    plt.savefig(os.path.join(path, 'cum_input'), bbox_inches='tight')


def plot_vmem(mem_layers, spikes_layers, labels, path, T, dt, title='V_mem'):
    num_layers = len(mem_layers)
    f, ax = plt.subplots(num_layers, 1, sharex=True, sharey=True)
    f.set_figheight(4 * num_layers)
    f.set_figwidth(10)
    ax[-1].set_xlabel('Simulation time')
    for i in range(num_layers):
        ax[i].plot(np.arange(0, T, dt), mem_layers[i], label='V_mem')
        ax[i].plot([s * dt for s in spikes_layers[i]], np.ones_like(spikes_layers[i]), '.', label='spikes')
        ax[i].set_title(labels[i])
        ax[i].set_ylabel('V')
        ax[i].legend()
    f.savefig(os.path.join(path, title), bbox_inches='tight')
