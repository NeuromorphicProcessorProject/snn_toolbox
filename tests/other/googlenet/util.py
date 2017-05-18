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


class ExpResults:
    def __init__(self, dirname, label, marker='.', color1='b', color5='r',
                 markersize=4, scale=1):
        self.dirname = os.path.join(dirname, 'log_vars')
        self.label = label
        self.marker = marker
        self.color1 = color1
        self.color5 = color5
        self.markersize = markersize
        self.scale = scale
        self.time = None
        self.mean_computations_t = self.std_computations_t = None
        self.e1_mean = self.e1_std = self.e5_mean = self.e5_std = None
        self.num_samples = None
        self.e1_confidence95 = self.e5_confidence95 = None
        self.op1_0 = self.op1_1 = self.op1_2 = None
        self.op5_0 = self.op5_1 = self.op5_2 = None
        self.e1_0 = self.e1_1 = self.e1_2 = None
        self.e5_0 = self.e5_1 = self.e5_2 = None
        self.e1_ann = self.e5_ann = None
        self.e1_confidence95_ann = self.e5_confidence95_ann = None
        self.e1_std_ann = self.e5_std_ann = None
        self.operations_ann = None
        self.e1_optimal = self.op1_optimal = None
        self.e5_optimal = self.op5_optimal = None
        self.set_spikestats()

    def set_spikestats(self):
        from snntoolbox.core.util import wilson_score

        num_batches = len(os.listdir(self.dirname))
        if num_batches == 0:
            return
        batch_size, num_timesteps = np.load(os.path.join(
            self.dirname, '0.npz'))['top1err_b_t'].shape
        self.time = np.arange(num_timesteps)
        self.num_samples = num_batches * batch_size
        e1 = np.empty((self.num_samples, num_timesteps))
        e5 = np.empty((self.num_samples, num_timesteps))

        # Load operation count
        operations_d_t = np.empty((self.num_samples, num_timesteps))
        for batch_idx in range(num_batches):
            operations_d_t[batch_idx*batch_size:(batch_idx+1)*batch_size] = \
                np.load(os.path.join(self.dirname, str(batch_idx) + '.npz'))[
                    'operations_b_t'] / self.scale
        self.mean_computations_t = np.mean(operations_d_t, 0)
        self.std_computations_t = np.std(operations_d_t, 0)

        # Load error
        for batch_idx in range(num_batches):
            e1[batch_idx * batch_size: (batch_idx + 1) * batch_size] = \
                np.multiply(100, np.load(os.path.join(
                    self.dirname, str(batch_idx) + '.npz'))['top1err_b_t'])
            e5[batch_idx * batch_size: (batch_idx + 1) * batch_size] = \
                np.multiply(100, np.load(os.path.join(
                    self.dirname, str(batch_idx) + '.npz'))['top5err_b_t'])

        self.operations_ann = float(np.load(os.path.join(self.dirname, str(
            num_batches - 1) + '.npz'))['operations_ann'] / self.scale)
        self.e1_ann = float(np.load(os.path.join(
            self.dirname, str(num_batches - 1) + '.npz'))['top1err_ann']) * 100
        self.e5_ann = float(np.load(os.path.join(
            self.dirname, str(num_batches - 1) + '.npz'))['top5err_ann']) * 100

        # Averaged across samples, shape (1, num_timesteps)
        self.e1_mean = np.mean(e1, axis=0)
        self.e1_std = np.std(e1, axis=0)
        self.e5_mean = np.mean(e5, axis=0)
        self.e5_std = np.std(e5, axis=0)
        self.e1_confidence95 = np.array([wilson_score(1-e/100, self.num_samples)
                                         for e in self.e1_mean]) * 100
        self.e5_confidence95 = np.array([wilson_score(1-e/100, self.num_samples)
                                         for e in self.e5_mean]) * 100

        # Get the operation count at which the error is minimal or 1 % above the
        # min.
        self.e1_0 = min(self.e1_mean)
        self.op1_0 = get_op_at_err(self.mean_computations_t, self.e1_mean,
                                   self.e1_0)
        self.e1_1 = min(self.e1_mean) + 1
        self.op1_1 = get_op_at_err(self.mean_computations_t, self.e1_mean,
                                   self.e1_1)
        self.e1_2 = get_err_at_op(self.e1_mean, self.mean_computations_t,
                                  self.operations_ann)
        self.op1_2 = get_op_at_err(self.mean_computations_t, self.e1_mean,
                                   self.e1_2)
        self.e5_0 = min(self.e5_mean)
        self.op5_0 = get_op_at_err(self.mean_computations_t, self.e5_mean,
                                   self.e5_0)
        self.e5_1 = min(self.e5_mean) + 1
        self.op5_1 = get_op_at_err(self.mean_computations_t, self.e5_mean,
                                   self.e5_1)
        self.op5_1 = get_op_at_err(self.mean_computations_t, self.e5_mean,
                                   self.e5_1)
        self.e5_2 = get_err_at_op(self.e5_mean, self.mean_computations_t,
                                  self.operations_ann)
        self.op5_2 = get_op_at_err(self.mean_computations_t, self.e5_mean,
                                   self.e5_2)

        self.e1_std_ann = get_std(self.e1_ann)
        self.e5_std_ann = get_std(self.e5_ann)
        self.e1_confidence95_ann = wilson_score(1 - self.e1_ann / 100,
                                                self.num_samples) * 100
        self.e5_confidence95_ann = wilson_score(1 - self.e5_ann / 100,
                                                self.num_samples) * 100

        self.e1_optimal, self.op1_optimal = get_minimal_err_and_op(
            self.mean_computations_t, self.e1_mean)

        self.e5_optimal, self.op5_optimal = get_minimal_err_and_op(
            self.mean_computations_t, self.e5_mean)

def get_std(err):
    return np.sqrt(err * (100 - err))


def get_op_at_err(ops_snn, err_snn, err_ann):
    t = np.where(err_snn <= err_ann)[0][0]
    return ops_snn[t]


def get_err_at_op(err_snn, ops_snn, ops_ann):
    if ops_ann > max(ops_snn):
        ops_ann = max(ops_snn)
    t = np.where(ops_snn >= ops_ann)[0][0]
    return err_snn[t]


def get_minimal_err_and_op(ops_t, err_t):
    aa = ops_t / max(ops_t)
    bb = err_t / 100
    c = [np.sqrt(a*a + b*b) for a, b in zip(aa, bb)]
    t = np.argmin(c)
    return err_t[t], ops_t[t]
