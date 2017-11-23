# -*- coding: utf-8 -*-

"""
Tools to load DVS sequence, preprocess it, and create batches of event-frames
for use in a time-stepped simulator.
"""

import os
import numpy as np
from snntoolbox.datasets.utils import to_categorical


class DVSIterator(object):
    def __init__(self, dataset_path, batch_shape, frame_width,
                 num_events_per_sample, chip_size, target_size=None,
                 label_dict=None):
        self.dataset_path = dataset_path
        self.batch_shape = batch_shape
        self.batch_size = batch_shape[0]
        self.frame_width = frame_width
        self.batch_idx = 0
        self.chip_size = chip_size
        self.target_size = target_size
        self.num_events_of_sample = 0
        self.dvs_sample_idx = -1
        self.num_events_per_sample = num_events_per_sample
        self.num_events_per_batch = self.batch_size * num_events_per_sample
        self.x_b_xaddr = self.x_b_yaddr = self.x_b_ts = self.y_b = None
        self.frames_from_sequence = None

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

    def next_sequence(self):
        """Get a new event sequence from disk, and update the counters."""

        # Increment number of samples loaded by DVSIterator.
        self.dvs_sample_idx += 1

        # Test if there are aedat files left to load.
        if self.dvs_sample_idx == len(self.filenames):
            raise StopIteration()

        # Load new sequence.
        filepath = os.path.join(self.dataset_path,
                                self.filenames[self.dvs_sample_idx])
        event_sequence = load_event_sequence(filepath, (239, 179))

        # Update statistics of current sequence.
        self.num_events_of_sample = len(event_sequence['x'])
        print("Total number of events of this sample: {}.".format(
            self.num_events_of_sample))
        print("Number of batches: {:d}.".format(
            int(self.num_events_of_sample / self.num_events_per_batch)))

        # Reset batch index, because new sequence will be used to generate new
        # batches.
        self.batch_idx = 0

        return event_sequence

    def next_sequence_batch(self):
        """
        Get a new batch of event sequences by chopping a long sequence into
        pieces.
        """

        # Load new sequence if all events of current sequence have been used.
        event_sequence = None
        if self.num_events_of_sample <= \
                self.num_events_per_batch * (self.batch_idx + 1):
            event_sequence = self.next_sequence()

            # Get class label, which is the same for all events in a sequence.
            self.y_b = np.broadcast_to(to_categorical(
                [self.labels[self.dvs_sample_idx]], self.num_classes),
                (self.batch_size, self.num_classes))

            # Generate frames from events.
            self.frames_from_sequence = get_frames_from_sequence(
                event_sequence['x'], event_sequence['y'],
                self.num_events_per_sample, self.chip_size, self.target_size)

        # From the current event sequence, extract the next bunch of events and
        # stack them as a batch of small sequences.
        self.x_b_xaddr, self.x_b_yaddr, self.x_b_ts = extract_batch(
            event_sequence['x'], event_sequence['y'], event_sequence['ts'],
            self.batch_size, self.batch_idx, self.num_events_per_sample,
            self.chip_size, self.target_size)

        self.batch_idx += 1

        return self.x_b_xaddr, self.x_b_yaddr, self.x_b_ts, self.y_b

    def next_eventframe_batch(self):
        return next_eventframe_batch(
            self.x_b_xaddr, self.x_b_yaddr, self.x_b_ts, self.batch_shape,
            self.frame_width)

    def get_frame_batch(self):
        event_idxs = range(self.num_events_per_batch * self.batch_idx,
                           self.num_events_per_batch * (self.batch_idx + 1))
        return self.frames_from_sequence[event_idxs]


def extract_batch(xaddr, yaddr, timestamps, batch_size, batch_idx,
                  num_events_per_sample, chip_size, target_size=None):
    """Transform a one-dimensional sequence of AER-events into a batch.

    Parameters
    ----------

    xaddr: ndarray
    yaddr: ndarray
    timestamps: ndarray
    batch_size: int
    batch_idx: int
    num_events_per_sample: int
    chip_size: tuple[int]
    target_size: Optional[tuple[int]]

    Returns
    -------

    x_b_xaddr, x_b_yaddr, x_b_ts: tuple[ndarray]
        Batch of event sequences.

    """

    from collections import deque

    if target_size is None:
        target_size = chip_size
        scale = None
    else:
        scale = [np.true_divide((t - 1), (c - 1)) for t, c in zip(target_size,
                                                                  chip_size)]

    x_b_xaddr = [deque() for _ in range(batch_size)]
    x_b_yaddr = [deque() for _ in range(batch_size)]
    x_b_ts = [deque() for _ in range(batch_size)]

    print("Extracting batch of samples à {} events from DVS sequence..."
          "".format(num_events_per_sample))

    for sample_idx in range(batch_size):
        start_event = num_events_per_sample * batch_size * batch_idx + \
                      num_events_per_sample * sample_idx
        event_idxs = range(start_event, start_event + num_events_per_sample)
        event_sums = np.zeros(target_size, 'int32')
        xaddr_sub = []
        yaddr_sub = []
        for x, y in zip(xaddr[event_idxs], yaddr[event_idxs]):
            if scale is not None:
                # Subsample from 240x180 to e.g. 64x64
                x = int(x * scale[0])
                y = int(y * scale[1])
            # Count event at subsampled location (x and y axes are swapped)
            event_sums[y, x] += 1
            xaddr_sub.append(x)
            yaddr_sub.append(y)

        # Compute standard deviation of event-sum distribution after removing
        # zeros
        sigma = np.std(event_sums[np.nonzero(event_sums)])

        # Clip number of events per pixel to three-sigma
        np.clip(event_sums, 0, 3 * sigma, event_sums)

        print("Discarded {} events during 3-sigma standardization.".format(
            num_events_per_sample - np.sum(event_sums)))

        for x, y, ts in zip(xaddr_sub, yaddr_sub, timestamps[event_idxs]):
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

    timestamps: ndarray
    xaddr: ndarray
    yaddr: ndarray
    pol: ndarray
    x_max: Optional[int]
    y_max: Optional[int]

    Returns
    -------

    timestamps: ndarray
    xaddr: ndarray
    yaddr: ndarray
    pol: ndarray
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


def load_event_sequence(filename, xyrange=None):
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

    dvs_sequence: dict[ndarray]
        Dictionary consisting of the following items:

            - x: np.array
                The x-addresses.
            - y: np.array
                The y-addresses.
            - ts: np.array
                The timestamps.

    """

    from snntoolbox.datasets.aedat import ImportAedat

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

    return {'x': xaddr, 'y': yaddr, 'ts': timestamps}


def get_binary_frame(xaddr, yaddr, timestamps, shape, frame_width):
    """
    Put events from event sequence into a shallow frame of at most one event per
    pixel. Stop if the time between the current and the oldest event exceeds
    ``frame_width``. Note that the events that have been added to the binary
    frame are removed from the input sequence!

    Parameters
    ----------

    xaddr: collections.deque
    yaddr: collections.deque
    timestamps: collections.deque
    shape: tuple
        Include channel dimension even for gray-scale images, e.g. (1, 64, 64).
    frame_width: int

    Returns
    -------

    binary_frame: ndarray
    """

    # Allocate output array.
    binary_frame = np.zeros(shape)

    # Buffer event sequence because we will be removing elements from original
    # list:
    xaddr_sample = list(xaddr)
    yaddr_sample = list(yaddr)
    ts_sample = list(timestamps)

    # Need first timestamp of current event sequence to determine when to stop
    # adding events.
    first_ts_of_frame = ts_sample[0] if ts_sample else 0

    # Put events from event sequence buffer into frame, if pixel location is not
    # occupied yet.
    for x, y, ts in zip(xaddr_sample, yaddr_sample, ts_sample):
        if binary_frame[0:, y, x] == 0:
            binary_frame[0:, y, x] = 1
            # Can't use .popleft()
            xaddr.remove(x)
            yaddr.remove(y)
            timestamps.remove(ts)
        # Start next frame if width of frame exceeds time limit.
        if ts - first_ts_of_frame > frame_width:
            break

    return binary_frame


def get_eventframe_sequence(xaddr, yaddr, timestamps, shape, frame_width):
    """
    Given a single sequence of x-y-ts events, generate a sequence of binary
    event frames.
    """

    inp = []

    while len(xaddr) > 0:
        inp.append(get_binary_frame(xaddr, yaddr, timestamps, shape,
                                    frame_width))

    return np.stack(inp, -1)


def next_eventframe_batch(x_b_xaddr, x_b_yaddr, x_b_ts, shape, frame_width):
    """
    Given a batch of x-y-ts event sequences, generate a batch of binary event
    frames that can be used in a time-stepped simulator.
    """

    # Allocate output array.
    input_b_l = np.zeros(shape, 'float32')

    # Generate each frame in batch sequentially.
    for sample_idx in range(shape[0]):
        input_b_l[sample_idx] = get_binary_frame(
            x_b_xaddr[sample_idx], x_b_yaddr[sample_idx], x_b_ts[sample_idx],
            shape[1:], frame_width)

    return input_b_l


def get_frames_from_sequence(xaddr, yaddr, num_events_per_frame, chip_size,
                             target_size=None):
    """
    Extract ``num_events_per_frame`` events from a one-dimensional sequence of
    AER-events. The events are spatially subsampled to ``target_size``, and
    standardized to [0, 1] using 3-sigma normalization. The resulting events
    are binned into a frame. The function operates on the events in
    ``xaddr`` etc sequentially until all are processed into frames.
    """

    if target_size is None:
        target_size = chip_size
        scale = None
    else:
        scale = [np.true_divide((t - 1), (c - 1)) for t, c in zip(target_size,
                                                                  chip_size)]

    num_frames = int(len(xaddr) / num_events_per_frame)
    frames = np.zeros([num_frames] + list(target_size), 'float32')

    print("Extracting {} frames from DVS event sequence.".format(num_frames))

    # Iterate for as long as there are events in the sequence.
    for sample_idx in range(num_frames):
        event_idxs = range(num_events_per_frame * sample_idx,
                           num_events_per_frame * (sample_idx + 1))

        # Loop over ``num_events_per_frame`` events
        for x, y in zip(xaddr[event_idxs], yaddr[event_idxs]):
            if scale is not None:
                # Subsample from 240x180 to e.g. 64x64
                x = int(x * scale[0])
                y = int(y * scale[1])
            # Count event at subsampled location (x and y axes are swapped)
            frames[sample_idx, y, x] += 1

        # Compute standard deviation of event-sum distribution after removing
        # zeros
        sample = frames[sample_idx]
        sigma = np.std(sample[np.nonzero(sample)])

        # Clip number of events per pixel to three-sigma
        frames[sample_idx] = np.clip(sample, 0, 3 * sigma)

    return frames / 255.
