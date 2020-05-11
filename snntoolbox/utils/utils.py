"""General utility functions on project-level.

@author: rbodo
"""

import importlib
import pkgutil

import json
import numpy as np
import os
import sys
import tempfile
import tensorflow as tf
from tensorflow import keras


def get_range(start=0.0, stop=1.0, num=5, method='linear'):
    """Return a range of parameter values.

    Convenience function. For more flexibility, use ``numpy.linspace``,
    ``numpy.logspace``, ``numpy.random.random_sample`` directly.

    Parameters
    ----------

    start: float
        The starting value of the sequence
    stop: float
        End value of the sequence.
    num: int
        Number of samples to generate. Must be non-negative.
    method: str
        The sequence will be computed on either a linear, logarithmic or random
        grid.

    Returns
    -------

    samples: np.array
        There are ``num`` samples in the closed interval [start, stop].
    """

    methods = {'linear', 'log', 'random'}
    assert method in methods, "Specified grid-search method {} not supported.\
        Choose among {}".format(method, methods)
    assert start < stop, "Start must be smaller than stop."
    assert num > 0 and isinstance(num, int), \
        "Number of samples must be unsigned int."
    if method == 'linear':
        return np.linspace(start, stop, num)
    if method == 'log':
        return np.logspace(start, stop, num, endpoint=False)
    if method == 'random':
        return np.random.random_sample(num) * (stop - start) + start


def confirm_overwrite(filepath):
    """
    If config.get('output', 'overwrite')==False and the file exists, ask user
    if it should be overwritten.
    """

    if os.path.isfile(filepath):
        overwrite = input("[WARNING] {} already exists - ".format(filepath) +
                          "overwrite? [y/n]")
        while overwrite not in ['y', 'n']:
            overwrite = input("Enter 'y' (overwrite) or 'n' (cancel).")
        return overwrite == 'y'
    return True


def to_json(data, path):
    """Write ``data`` dictionary to ``path``.

    A :py:exc:`TypeError` is raised if objects in ``data`` are not JSON
    serializable.
    """

    def get_json_type(obj):
        """Get type of object to check if JSON serializable.

        Parameters
        ----------

        obj: object

        Raises
        ------

        TypeError

        Returns
        -------

        : Union(string, Any)
        """

        if type(obj).__module__ == np.__name__:
            # noinspection PyUnresolvedReferences
            return obj.item()

        # if obj is a python 'type'
        if type(obj).__name__ == type.__name__:
            return obj.__name__

        raise TypeError("{} not JSON serializable".format(type(obj).__name__))

    json.dump(data, open(path, str('w')), default=get_json_type)


def import_helpers(filepath, config):
    """Import a module with helper functions from ``filepath``.

    Parameters
    ----------

    filepath: str
        Filename or relative or absolute path of module to import. If only
        the filename is given, module is assumed to be in current working
        directory (``config.get('paths', 'path_wd')``). Non-absolute paths are
        taken relative to working dir.
    config: configparser.ConfigParser
        Settings.

    Returns
    -------

    :
        Module with helper functions.

    """

    path, filename = get_abs_path(filepath, config)

    return import_script(path, filename)


def get_abs_path(filepath, config):
    """Get an absolute path, possibly using current toolbox working dir.

    Parameters
    ----------

    filepath: str
        Filename or relative or absolute path. If only the filename is given,
        file is assumed to be in current working directory
        (``config.get('paths', 'path_wd')``). Non-absolute paths are
        interpreted relative to working dir.
    config: configparser.ConfigParser
        Settings.

    Returns
    -------

    path: str
        Absolute path to file.

    """

    path, filename = os.path.split(filepath)
    if path == '':
        path = config.get('paths', 'path_wd')
    elif not os.path.isabs(path):
        path = os.path.abspath(os.path.join(config.get('paths', 'path_wd'),
                                            path))
    return path, filename


def import_script(path, filename):
    """Import python script independently from python version.

    Parameters
    ----------

    path: string
        Path to directory where to load script from.

    filename: string
        Name of script file.
    """

    filepath = os.path.join(path, filename + '.py')

    v = sys.version_info
    if v >= (3, 5):
        import importlib.util
        spec = importlib.util.spec_from_file_location(filename, filepath)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    elif v >= (3, 3):
        # noinspection PyCompatibility,PyUnresolvedReferences
        from importlib.machinery import SourceFileLoader
        mod = SourceFileLoader(filename, filepath).load_module()
    else:
        # noinspection PyDeprecation
        import imp
        # noinspection PyDeprecation
        mod = imp.load_source(filename, filepath)
    return mod


def binary_tanh(x):
    """Round a float to -1 or 1.

    Parameters
    ----------

    x: float

    Returns
    -------

    : int
        Integer in {-1, 1}
    """

    return keras.backend.sign(x)


def binary_sigmoid(x):
    """Round a float to 0 or 1.

    Parameters
    ----------

    x: float

    Returns
    -------

    : int
        Integer in {0, 1}
    """

    return keras.backend.round(hard_sigmoid(x))


def hard_sigmoid(x):
    """

    Parameters
    ----------

    x :

    Returns
    -------

    """

    return keras.backend.clip((x + 1.) / 2., 0, 1)


def binarize_var(w, h=1., deterministic=True):
    """Binarize shared variable.

    Parameters
    ----------

    w: keras.backend.Variable
        Weights.
    h: float
        Values are round to ``+/-h``.
    deterministic: bool
        Whether to apply deterministic rounding.

    Returns
    -------

    w: keras.backend.variable
        The binarized weights.
    """

    # [-1, 1] -> [0, 1]
    wb = hard_sigmoid(w / h)

    # Deterministic / stochastic rounding
    wb = keras.backend.round(wb) if deterministic \
        else keras.backend.cast_to_floatx(np.random.binomial(1, wb))

    # {0, 1} -> {-1, 1}
    wb = keras.backend.cast_to_floatx(keras.backend.switch(wb, h, -h))

    return keras.backend.cast_to_floatx(wb)


def binarize(w, h=1., deterministic=True):
    """Binarize weights.

    Parameters
    ----------

    w: ndarray
        Weights.
    h: float
        Values are round to ``+/-h``.
    deterministic: bool
        Whether to apply deterministic rounding.

    Returns
    -------

    : ndarray
        The binarized weights.
    """

    # [-1, 1] -> [0, 1]
    wb = np.clip((np.add(np.true_divide(w, h), 1.)) / 2., 0, 1)

    # Deterministic / stochastic rounding
    wb = np.round(wb) if deterministic else np.random.binomial(1, wb)

    # {0, 1} -> {-1, 1}
    wb[wb != 0] = h
    wb[wb == 0] = -h

    return np.asarray(wb, np.float32)


def reduce_precision(x, m, f):
    """Reduces precision of ``x`` to format ``Qm.f``.

    Parameters
    ----------

    x : ndarray
        Input data.
    m : int
        Number of integer bits.
    f : int
        Number of fractional bits.

    Returns
    -------

    x_lp : ndarray
        The input data with reduced precision.

    """
    n = 2 << f - 1
    maxval = (2 << m - 1) - 1.0 / n
    return np.clip(np.true_divide(np.round(x * n), n), -maxval, maxval)


def reduce_precision_var(x, m, f):
    """Reduces precision of ``x`` to format ``Qm.f``.

    Parameters
    ----------

    x : keras.backend.variable
        Input data.
    m : int
        Number of integer bits.
    f : int
        Number of fractional bits.

    Returns
    -------

    x_lp : keras.backend.variable
        The input data with reduced precision.

    """
    n = 2 << f - 1
    maxval = (2 << m - 1) - 1.0 / n
    return keras.backend.clip(keras.backend.round(x * n) / n, -maxval, maxval)


def quantized_relu(x, m, f):
    """
    Rectified linear unit activation function with precision of ``x`` reduced
    to fixed point format ``Qm.f``.

    Parameters
    ----------

    x : keras.backend.variable
        Input data.
    m : int
        Number of integer bits.
    f : int
        Number of fractional bits.

    Returns
    -------

    x_lp : keras.backend.variable
        The input data with reduced precision.

    """
    return keras.backend.relu(reduce_precision_var(x, m, f))


class LimitedReLU(keras.layers.ReLU):
    def __init__(self, cfg):
        super(LimitedReLU, self).__init__(**cfg)
        self.__name__ = '{}_{}_{}_LimitedReLU'.format(
            self.negative_slope, self.max_value, self.threshold)

    def get_cfg(self):
        return self.get_config()

    def set_cfg(self, cfg):
        self.__init__(cfg)

    def __call__(self, *args, **kwargs):
        return super(LimitedReLU, self).call(args[0])


class ClampedReLU:
    """
    Rectified linear unit activation function where values in ``x`` below
    ``threshold`` are clamped to 0, and values above ``max_value`` are clipped
    to ``max_value``.

    Attributes
    ----------

    threshold : Optional[float]
    max_value : Optional[float]

    """

    def __init__(self, threshold=0.1, max_value=None):  # Todo: Change defaults
        self.threshold = threshold
        self.max_value = max_value
        self.__name__ = 'clamped_relu_{}_{}'.format(self.threshold,
                                                    self.max_value)

    def __call__(self, *args, **kwargs):
        x = keras.backend.relu(args[0], max_value=self.max_value)
        return tf.where(keras.backend.less(x, self.threshold),
                        keras.backend.zeros_like(x), x)


class NoisySoftplus:
    def __init__(self, k=0.17, sigma=1):
        self.k = k
        self.sigma = sigma
        self.__name__ = 'noisy_softplus_{}_{}'.format(self.k, self.sigma)
                
    def __call__(self, *args, **kwargs):
        return self.k * self.sigma * keras.backend.softplus(
            args[0] / (self.k * self.sigma))


def wilson_score(p, n):
    """Confidence interval of a binomial distribution.

    See https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval.

    Parameters
    ----------

    p: float
        The proportion of successes in ``n`` experiments.
    n: int
        The number of Bernoulli-trials (sample size).

    Returns
    -------

    The confidence interval.
    """

    if n == 0:
        return 0

    # Quantile z of a standard normal distribution, for the error quantile a:
    z = 1.96  # 1.44 for a == 85%, 1.96 for a == 95%
    return (z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / (1 + z * z / n)


def extract_label(label):
    """Get the layer number, name and shape from a string.

    Parameters
    ----------

    label: str
        Specifies both the layer type, index and shape, e.g.
        ``'03Conv2D_3x32x32'``.

    Returns
    -------

    : tuple[int, str, tuple]
        - layer_num: The index of the layer in the network.
        - name: The type of the layer.
        - shape: The shape of the layer
    """

    label = label.split('_')
    layer_num = None
    for i in range(max(4, len(label) - 2)):
        if label[0][:i].isdigit():
            layer_num = int(label[0][:i])
    name = ''.join(s for s in label[0] if not s.isdigit())
    if name[-1] == 'D':
        name = name[:-1]
    if len(label) > 1:
        shape = tuple([int(s) for s in label[-1].split('x')])
    else:
        shape = ()
    return layer_num, name, shape


def in_top_k(predictions, targets, k):
    """Returns whether the ``targets`` are in the top ``k`` ``predictions``.

    # Arguments
        predictions: A tensor of shape batch_size x classes and type float32.
        targets: A tensor of shape batch_size and type int32 or int64.
        k: An int, number of top elements to consider.

    # Returns
        A tensor of shape batch_size and type int. output_i is 1 if
        targets_i is within top-k values of predictions_i
    """

    predictions_top_k = np.argsort(predictions)[:, -k:]
    return np.array([np.equal(p, t).any() for p, t in zip(predictions_top_k,
                                                          targets)])


def top_k_categorical_accuracy(y_true, y_pred, k=5):
    """

    Parameters
    ----------
    y_true :
    y_pred :
    k :

    Returns
    -------

    """

    return np.mean(in_top_k(y_pred, np.argmax(y_true, axis=-1), k))


def echo(text):
    """python 2 version of print(end='', flush=True)."""

    sys.stdout.write(u'{}'.format(text))
    sys.stdout.flush()


def to_list(x):
    """Normalize a list/tensor to a list.

    If a tensor is passed, returns a list of size 1 containing the tensor.
    """

    return x if type(x) is list else [x]


def apply_modifications(model, custom_objects=None):
    """Applies modifications to the model layers to create a new Graph. For
    example, simply changing ``model.layers[idx].activation = new activation``
    does not change the graph. The entire graph needs to be updated with
    modified inbound and outbound tensors because of change in layer building
    function.

    Parameters
    ----------

        model: keras.models.Model

        custom_objects: dict

    Returns
    -------

        The modified model with changes applied. Does not mutate the original
        ``model``.
    """

    # The strategy is to save the modified model and load it back. This is done
    # because setting the activation in a Keras layer doesnt actually change
    # the graph. We have to iterate the entire graph and change the layer
    # inbound and outbound nodes with modified tensors. This is doubly
    # complicated in Keras 2.x since multiple inbound and outbound nodes are
    # allowed with the Graph API.

    # Taken from
    # https://github.com/raghakot/keras-vis/blob/master/vis/utils/utils.py

    # noinspection PyProtectedMember
    model_path = os.path.join(tempfile.gettempdir(),
                              next(tempfile._get_candidate_names()) + '.h5')
    try:
        model.save(model_path)
        return keras.models.load_model(model_path, custom_objects)
    finally:
        os.remove(model_path)


def import_configparser():
    try:
        import configparser
    except ImportError:
        # noinspection PyPep8Naming
        import ConfigParser as configparser

    return configparser


def is_module_installed(mod):
    if sys.version_info[0] < 3:
        return pkgutil.find_loader(mod) is not None
    else:
        return importlib.util.find_spec(mod) is not None


def get_pearson_coefficients(spikerates_batch, activations_batch, max_rate):
    """
    Compute Pearson coefficients.

    Parameters
    ----------

    spikerates_batch : 
    activations_batch :
    max_rate: float
        Highest spike rate.

    Returns
    -------
    
    co: list

    """

    co = []
    for layer_num in range(len(spikerates_batch)):
        c = []
        for sample in range(len(spikerates_batch[0][0])):
            s = spikerates_batch[layer_num][0][sample].flatten()
            a = activations_batch[layer_num][0][sample].flatten()
            if layer_num < len(spikerates_batch) - 1:
                # Remove points at origin and saturated units, except for
                # output layer (has too few units and gets activation of 1
                # because of softmax).
                ss = []
                aa = []
                for sss, aaa in zip(s, a):
                    if (sss > 0 or aaa > 0) and aaa < max_rate:
                        ss.append(sss)
                        aa.append(aaa)
                s = ss
                a = aa
            c.append(np.corrcoef(s, a)[0, 1])
        co.append(c)

    return co
