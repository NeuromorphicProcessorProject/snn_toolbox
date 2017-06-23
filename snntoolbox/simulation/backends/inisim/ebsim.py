# coding=utf-8

"""Basic event-driven simulator."""

# TODO: Implement biases

import os
import sys
import h5py
import numpy as np

if sys.version_info[0] < 3:
    # noinspection PyCompatibility
    from Queue import Queue
else:
    # noinspection PyCompatibility
    from queue import Queue

queue = Queue()


def create_connection_file(path):
    """
    
    Parameters
    ----------
    path : 

    Returns
    -------

    """

    f = h5py.File(os.path.join(path, 'connections'), 'w')
    index_group = f.create_group('fanout_indices')
    weight_group = f.create_group('fanout_weights')
    # 'int32' is sufficient for up to 4'294'967'296 neurons
    for neuron in network:
        index_group.create_dataset(str(neuron), data=fanout_indices[neuron])
        weight_group.create_dataset(str(neuron), data=fanout_weights[neuron])


class Event:
    """
    Event
    """

    def __init__(self, source_index, timestamp):
        self.source_index = source_index
        self.timestamp = timestamp


class Population:
    """
    Population
    """

    def __init__(self, size, name):
        self.v = np.zeros(size)
        self.name = name


def reset_mem(method='reset_to_zero'):
    """

    Parameters
    ----------
    method : 
    """

    if method == 'reset_by_subtraction':
        v -= settings['v_thresh']
    else:
        v = 0


def update_neuron(idx, input, t):
    """

    Parameters
    ----------
    idx : 
    input : 
    t : 
    """

    v[idx] += input
    if v[idx] >= settings['v_thresh']:
        queue.put(Event(idx, t))
        reset_mem()


def run():
    """
    Run
    """

    f = h5py.File(os.path.join(path, 'connections.h5'), 'r')
    fanout_index_group = f['fanout_indices']
    fanout_weight_group = f['fanout_weights']
    t = 0
    while True:
        event = queue.get()
        if event is None or t >= settings['duration']:
            break
        t = event.timestamp + settings['dt']  # synaptic delay
        source_index_string = str(event.source_index)
        fanout_indices = fanout_index_group[source_index_string]
        fanout_weights = fanout_weight_group[source_index_string]
        for i, w in zip(fanout_indices, fanout_weights):
            update_neuron(i, w, t)
