import os
import numpy as np
import matplotlib.pyplot as plt
import moviepy.editor as mpy
from moviepy.video.io.bindings import mplfig_to_npimage


def load_data_for_video(path, classification_duration, simulation_duration,
                        video_duration, num_samples, num_layers, class_idx_dict,
                        undo_preprocessing=None):

    if undo_preprocessing is None:
        def undo_preprocessing(x):
            return x

    input_images = []
    layer_labels = []
    top5_labels_t = [[] for _ in range(5)]
    top5_spikecounts_t = [[] for _ in range(5)]
    spiketrains_n = [[] for _ in range(num_layers)]
    true_labels_t = []
    for sample_idx in range(num_samples):
        log_vars = np.load(os.path.join(path, 'log_vars', str(sample_idx)
                                        + '.npz'))
        input_image_l = log_vars['input_image_b_l'][0]
        input_images.append(np.swapaxes(undo_preprocessing(input_image_l), 0,
                                        -1))

        spiketrains_n_b_l_t = log_vars['spiketrains_n_b_l_t']
        for layer_idx, spiketrains_b_l_t in enumerate(spiketrains_n_b_l_t):
            layer_labels.append(spiketrains_b_l_t[1])
            spiketimes = np.reshape(spiketrains_b_l_t[0],
                                    (-1, simulation_duration))[:100]
            spiketimes[spiketimes > 0] += sample_idx * simulation_duration
            spiketrains_n[layer_idx].append(spiketimes)

        spikes_l_t = np.greater(spiketrains_n_b_l_t[-1][0][0], 0)
        spikecounts_l_t = np.cumsum(spikes_l_t, axis=1)
        top5_classes_t = np.argsort(spikecounts_l_t, axis=0)[-5:]
        for i, class_idx_t in enumerate(top5_classes_t):
            top5_spikecounts_t[i] += [spikecounts_l_t[c, t] for t, c in
                                      enumerate(class_idx_t)]
            top5_labels_t[i] += [class_idx_dict[str(c)][1] for c in class_idx_t]

        true_label = class_idx_dict[str(log_vars['true_classes_b'][0])][1]
        true_labels_t += [true_label for _ in range(classification_duration)]

    input_images_t = np.empty(list(input_images[0].shape) + [video_duration])
    for i, input_image in enumerate(input_images):
        input_images_t[:, :, :, i * classification_duration:
                       (i + 1) * classification_duration] = \
            np.expand_dims(input_image, -1)

    spiketrains_n = [np.concatenate(s, 1) for s in spiketrains_n]

    return input_images_t, spiketrains_n, np.array(top5_labels_t),\
        np.array(top5_spikecounts_t), true_labels_t


def show_animated_raster_plots(spiketrains_n, classification_duration,
                               video_duration, path, dt=1e-3):

    num_layers = len(spiketrains_n)
    fig, ax = plt.subplots(num_layers, figsize=(60, 20))

    spiketrain_l = [[] for _ in range(num_layers)]
    y_l = [[] for _ in range(num_layers)]
    for j, s in enumerate(spiketrains_n):
        i = num_layers - j - 1
        for neuron, spiketrain in enumerate(s):
            spiketrain_l[i].append(spiketrain[spiketrain.nonzero()])
            y_l[i].append(np.ones_like(spiketrain_l[i][-1]) * neuron)
        spiketrain_l[i] = np.concatenate(spiketrain_l[i])
        y_l[i] = np.concatenate(y_l[i])
        ax[i].scatter(spiketrain_l[i], y_l[i], s=20)
        ax[i].get_xaxis().set_visible(False)
        ax[i].get_yaxis().set_ticks_position('none')
        ax[i].get_yaxis().set_ticklabels([])
        ax[i].set_ylabel('L {}'.format(j), fontsize=50)
        ax[i].spines["top"].set_visible(False)
        ax[i].spines["right"].set_visible(False)
        ax[i].spines["bottom"].set_visible(False)

    def make_frame(t):
        for axis in ax:
            axis.set(xlim=((t - 3 * classification_duration) * 1000, t * 1000))
        return mplfig_to_npimage(fig)

    animation = mpy.VideoClip(make_frame, duration=video_duration)
    animation.write_videofile(os.path.join(path, 'raster.mp4'), fps=1/dt)

    return animation


def show_input_image(images_t, classification_duration, video_duration, path):
    def make_frame(t):
        return images_t[:, :, :, int(t * 1000)]

    animation = mpy.VideoClip(make_frame, duration=video_duration)
    animation.write_videofile(os.path.join(path, 'images.mp4'),
                              fps=1/classification_duration)

    return animation


def show_labels(top5_labels_t, top5_spikecounts_t, true_labels_t,
                duration, path, dt=1e-3):
    fig, ax = plt.subplots(5, figsize=(10, 10))
    for i in range(len(ax)):
        ax[i].text(0, 0, '')
        ax[i].set_axis_off()

    def make_frame(t):
        min_font = 30
        max_font = 100
        t_int = int(t * 1000)
        top_confidence = top5_spikecounts_t[-1, t_int]
        for k in range(len(ax)):
            j = len(ax) - k - 1
            guessed_label = top5_labels_t[j, t_int]
            confidence = top5_spikecounts_t[j, t_int]
            color = 'green' if guessed_label == true_labels_t[t_int] else 'red'
            size = max(0, min_font + confidence - (top_confidence - max_font)) \
                if top_confidence >= max_font and j != len(ax) - 1 else \
                min(max_font, min_font + confidence)
            fontdict = {'color': color, 'size': size, 'family': 'sans-serif',
                        'weight': 'light'}
            ax[k].clear()
            ax[k].text(0, 0, guessed_label, fontdict=fontdict)
            ax[k].set_axis_off()
        return mplfig_to_npimage(fig)

    animation = mpy.VideoClip(make_frame, duration=duration)
    animation.write_videofile(os.path.join(path, 'labels.mp4'), fps=1/dt)

    return animation


def show_architecture(duration, path):
    animation = mpy.ImageClip(os.path.join(path, 'architecture.png'),
                              duration=duration)
    animation.write_videofile(os.path.join(path, 'architecture.mp4'),
                              fps=1/duration)
    return animation


def apply_title(clip, title):
    txt = mpy.TextClip(title, font='Purisa-Bold', fontsize=15).set_position(
        ('center', 'top')).set_duration(clip.duration)
    return mpy.CompositeVideoClip([clip, txt])


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


def get_rates(spiketimes, duration, dt, t_clamp=0, idx=0):
    # ``idx`` is the feature index.
    rates = [0.] * t_clamp
    count = 0
    t_idx = int(np.ceil(t_clamp/dt))
    for t in np.arange(t_clamp, duration, dt):
        count += int(spiketimes[idx][t_idx] > 0)
        rates.append(count/(t+dt-t_clamp))
        t_idx += 1
    return rates


def plot_spikerates(spiketimes_layers, labels, path, duration, dt,
                    target_activations=None, t_clamp=None, idx=0,
                    filename='spikerates'):
    plt.xlabel('Simulation time')
    plt.ylabel('Spike-rate')
    # plt.cm.plasma(np.linspace(0, 0.9, len(labels)))
    colors = ['blue', 'green', 'red']
    if t_clamp is None:
        t_clamp = len(spiketimes_layers) * [0]
    for i, spiketimes in enumerate(spiketimes_layers):
        plt.plot(np.arange(0, duration, dt),
                 get_rates(spiketimes, t_clamp[i], idx), label=labels[i],
                 color=colors[i])
        if target_activations:
            plt.plot([0, duration],
                     [target_activations[i], target_activations[i]],
                     label='{}_target'.format(labels[i]), color=colors[i])
    plt.legend(loc='center right')
    plt.savefig(os.path.join(path, filename), bbox_inches='tight')


def plot_cumsum(input_t, labels, path, duration, dt):
    plt.xlabel('Simulation time')
    plt.ylabel('Cumulated input to membrane potential')
    for i, inp in enumerate(input_t):
        plt.plot(np.arange(0, duration, dt), np.cumsum(inp), label=labels[i])
    plt.legend()
    plt.savefig(os.path.join(path, 'cum_input'), bbox_inches='tight')


def plot_vmem(mem_layers, spikes_layers, labels, path, duration, dt,
              title='V_mem'):
    num_layers = len(mem_layers)
    f, ax = plt.subplots(num_layers, 1, sharex=True, sharey=True)
    f.set_figheight(4 * num_layers)
    f.set_figwidth(10)
    ax[-1].set_xlabel('Simulation time')
    for i in range(num_layers):
        ax[i].plot(np.arange(0, duration, dt), mem_layers[i], label='V_mem')
        ax[i].plot([s * dt for s in spikes_layers[i]], np.ones_like(
            spikes_layers[i]), '.', label='spikes')
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
        from snntoolbox.utils.utils import wilson_score

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
                    'synaptic_operations_b_t'] / self.scale
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
        self.e1_1 = get_err_at_op(self.e1_mean, self.mean_computations_t,
                                  self.op1_1)
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
        self.e5_1 = get_err_at_op(self.e5_mean, self.mean_computations_t,
                                  self.op5_1)
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


def get_op_at_err(ops_t, err_t, err_ref):
    return ops_t[np.where(err_t <= err_ref)[0][0]]


def get_err_at_op(err_t, ops_t, ops_ref):
    if ops_ref > max(ops_t):
        ops_ref = max(ops_t)
    return err_t[np.where(ops_t >= ops_ref)[0][0]]


def get_minimal_err_and_op(ops_t, err_t):
    aa = ops_t / max(ops_t)
    bb = np.true_divide(err_t, 100)
    c = [np.sqrt(a*a + b*b) for a, b in zip(aa, bb)]
    t = np.argmin(c)
    return err_t[t], ops_t[t]
