# -*- coding: utf-8 -*-

"""
Tools to load DVS sequence, preprocess it, and create batches of event-frames
for use in a time-stepped simulator.
"""

import os
import numpy as np
from snntoolbox.io_utils.common import to_categorical


class DVSIterator(object):
    """

    Parameters
    ----------
    dataset_path :
    batch_size :
    scale:

    Returns
    -------

    """

    def __init__(self, dataset_path, batch_size, label_dict=None,
                 scale=None, num_events_per_sample=1000):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.batch_idx = 0
        self.scale = scale
        self.xaddr_sequence = None
        self.yaddr_sequence = None
        self.dvs_sample = None
        self.num_events_of_sample = 0
        self.dvs_sample_idx = -1
        self.num_events_per_sample = num_events_per_sample
        self.num_events_per_batch = batch_size * num_events_per_sample

        # Count the number of samples and classes
        classes = [subdir for subdir in sorted(os.listdir(dataset_path))
                   if os.path.isdir(os.path.join(dataset_path, subdir))]

        self.label_dict = dict(zip(classes, range(len(classes)))) \
            if not label_dict else label_dict
        self.num_classes = len(label_dict)
        assert self.num_classes == len(classes), \
            "The number of classes provided by label_dict {} does not match " \
            "the number of subdirectories found in dataset_path {}.".format(
                self.label_dict, self.dataset_path)

        self.filenames = []
        labels = []
        self.num_samples = 0
        for subdir in classes:
            for fname in sorted(os.listdir(os.path.join(dataset_path, subdir))):
                is_valid = False
                for extension in {'aedat'}:
                    if fname.lower().endswith('.' + extension):
                        is_valid = True
                        break
                if is_valid:
                    labels.append(self.label_dict[subdir])
                    self.filenames.append(os.path.join(subdir, fname))
                    self.num_samples += 1
        self.labels = np.array(labels, 'int32')
        print("Found {} samples belonging to {} classes.".format(
            self.num_samples, self.num_classes))

    def __next__(self):
        while self.num_events_per_batch * (self.batch_idx + 1) >= \
                self.num_events_of_sample:
            self.dvs_sample_idx += 1
            if self.dvs_sample_idx == len(self.filenames):
                raise StopIteration()
            filepath = os.path.join(self.dataset_path,
                                    self.filenames[self.dvs_sample_idx])
            self.dvs_sample = load_dvs_sequence(filepath, (239, 179))
            self.num_events_of_sample = len(self.dvs_sample[0])
            self.batch_idx = 0
            print("Total number of events of this sample: {}.".format(
                self.num_events_of_sample))
            print("Number of batches: {:d}.".format(
                int(self.num_events_of_sample / self.num_events_per_batch)))

        batch_idx = self.batch_idx
        self.batch_idx += 1

        xaddr_b, yaddr_b, ts_b = extract_batch(
            self.dvs_sample[0], self.dvs_sample[1], self.dvs_sample[2],
            self.batch_size, batch_idx, self.num_events_per_sample, self.scale)

        # Each sample in the batch has the same label because it is generated
        # from the same DVS sequence.
        truth_b = np.broadcast_to(to_categorical(
            [self.labels[self.dvs_sample_idx]], self.num_classes),
            (self.batch_size, self.num_classes))

        return xaddr_b, yaddr_b, ts_b, truth_b


def extract_batch(xaddr, yaddr, timestamps, batch_size, batch_idx=0,
                  num_events_per_sample=2000, scale=(239/63, 179/63)):
    """
    Transform a one-dimensional sequence of AER-events into a batch.

    :param xaddr: 
    :type xaddr: 
    :param yaddr: 
    :type yaddr: 
    :param timestamps: 
    :type timestamps: 
    :param batch_size: 
    :type batch_size: 
    :param batch_idx: 
    :type batch_idx: 
    :param num_events_per_sample: 
    :type num_events_per_sample: 
    :param scale: 
    :type scale: 
    :return: 
    :rtype: 
    """

    from collections import deque

    print("Extracting batch of samples à {} events from DVS sequence..."
          "".format(num_events_per_sample))

    x_b_xaddr = [deque() for _ in range(batch_size)]
    x_b_yaddr = [deque() for _ in range(batch_size)]
    x_b_ts = [deque() for _ in range(batch_size)]
    for sample_idx in range(batch_size):
        start_event = num_events_per_sample * batch_size * batch_idx + \
                      num_events_per_sample * sample_idx
        event_idxs = range(start_event, start_event + num_events_per_sample)
        event_sums = np.zeros((64, 64), 'int32')
        xaddr_sub = []
        yaddr_sub = []
        for x, y in zip(xaddr[event_idxs], yaddr[event_idxs]):
            if scale:
                # Subsample from 240x180 to e.g. 64x64
                x = int(x / scale[0])
                y = int(y / scale[1])
            event_sums[y, x] += 1
            xaddr_sub.append(x)
            yaddr_sub.append(y)
        sigma = np.std(event_sums[np.nonzero(event_sums)])
        # Clip number of events per pixel to three-sigma
        np.clip(event_sums, 0, 3*sigma, event_sums)
        print("Discarded {} events during 3-sigma standardization.".format(
            num_events_per_sample - np.sum(event_sums)))
        ts_sample = timestamps[event_idxs]
        for x, y, ts in zip(xaddr_sub, yaddr_sub, ts_sample):
            if event_sums[y, x] > 0:
                x_b_xaddr[sample_idx].append(x)
                x_b_yaddr[sample_idx].append(y)
                x_b_ts[sample_idx].append(ts)
                event_sums[y, x] -= 1

    return x_b_xaddr, x_b_yaddr, x_b_ts


def remove_outliers(timestamps, xaddr, yaddr, pol, x_max=239, y_max=179):
    """Remove outliers from DVS data.

    Parameters
    ----------
    timestamps :
    xaddr :
    yaddr :
    pol :
    x_max :
    y_max :

    Returns
    -------

    """

    len_orig = len(timestamps)
    xaddr_valid = np.where(np.array(xaddr) <= x_max)
    yaddr_valid = np.where(np.array(yaddr) <= y_max)
    xy_valid = np.intersect1d(xaddr_valid[0], yaddr_valid[0], True)
    xaddr = np.array(xaddr)[xy_valid]
    yaddr = np.array(yaddr)[xy_valid]
    timestamps = np.array(timestamps)[xy_valid]
    pol = np.array(pol)[xy_valid]
    num_outliers = len_orig - len(timestamps)
    if num_outliers:
        print("Removed {} outliers.".format(num_outliers))
    return timestamps, xaddr, yaddr, pol


def load_dvs_sequence(filename, xyrange=None):
    """
    Load a sequence of AER-events from an ``.aedat`` file and return three
    arrays containing the x, y addresses and the timestamps.
    If an ``xyrange`` is given, events outside this range are removed. 

    Parameters
    ----------

    filename: str
        Name of ``.aedat`` file to load.
    xyrange: tuple[int]
        Chip dimensions - 1, i.e. largest indices with zero-convention. 

    Returns
    -------

    xaddr: np.array
        The x-addresses.
    yaddr: np.array
        The y-addresses.
    timestamps: np.array
        The timestamps.

    """

    from snntoolbox.io_utils.AedatTools import ImportAedat

    print("Loading DVS sample {}...".format(filename))
    events = ImportAedat.import_aedat({'filePathAndName':
                                       filename})['data']['polarity']
    timestamps = events['timeStamp']
    xaddr = events['x']
    yaddr = events['y']
    pol = events['polarity']

    # Remove events with addresses outside valid range
    if xyrange:
        timestamps, xaddr, yaddr, pol = remove_outliers(
            timestamps, xaddr, yaddr, pol, xyrange[0], xyrange[1])

    xaddr = xyrange[0] - xaddr
    yaddr = xyrange[1] - yaddr

    return xaddr, yaddr, timestamps
