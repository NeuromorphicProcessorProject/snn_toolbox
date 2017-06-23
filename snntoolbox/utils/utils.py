"""General utility functions on project-level.

@author: rbodo
"""

import os

import keras
import numpy as np


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
    If config['output']['overwrite']==False and the file exists, ask user if it
    should be overwritten.
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

    import json

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
        directory (``config['paths']['path_wd']``). Non-absolute paths are taken
        relative to working dir.
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
        (``config['paths']['path_wd']``). Non-absolute paths are interpreted
        relative to working dir.
    config: configparser.ConfigParser
        Settings.

    Returns
    -------

    path: str
        Absolute path to file.

    """

    path, filename = os.path.split(filepath)
    if path == '':
        path = config['paths']['path_wd']
    elif not os.path.isabs(path):
        path = os.path.abspath(os.path.join(config['paths']['path_wd'], path))
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

    import sys

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
    w: np.array
        Weights.
    h: float
        Values are round to ``+/-h``.
    deterministic: bool
        Whether to apply deterministic rounding.

    Returns
    -------

    : np.array
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

    : np.array
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

    l = label.split('_')
    layer_num = None
    for i in range(max(4, len(l) - 2)):
        if l[0][:i].isnumeric():
            layer_num = int(l[0][:i])
    name = ''.join(s for s in l[0] if not s.isdigit())
    if name[-1] == 'D':
        name = name[:-1]
    if len(l) > 1:
        shape = tuple([int(s) for s in l[-1].split('x')])
    else:
        shape = ()
    return layer_num, name, shape


def in_top_k(predictions, targets, k):
    """Returns whether the ``targets`` are in the top ``k`` ``predictions``.

    # Arguments
        predictions: A tensor of shape batch_size x classess and type float32.
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

# python 2 can not handle the 'flush' keyword argument of python 3 print().
# Provide 'echo' function as an alias for
# "print with flush and without newline".
try:
    from functools import partial
    echo = partial(print, end='', flush=True)
    echo(u'')
except TypeError:
    # TypeError: 'flush' is an invalid keyword argument for this function
    import sys

    def echo(text):
        """python 2 version of print(end='', flush=True)."""
        sys.stdout.write(u'{0}'.format(text))
        sys.stdout.flush()


def to_list(x):
    """Normalize a list/tensor to a list.

    If a tensor is passed, returns a list of size 1 containing the tensor.
    """

    return x if type(x) is list else [x]
