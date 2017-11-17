# -*- coding: utf-8 -*-

"""
Tools to load DVS sequence, preprocess it, and create batches of event-frames
for use in a time-stepped simulator.
"""

import os
import numpy as np
from snntoolbox.datasets.utils import to_categorical


class DVSIterator(object):
    def __init__(self, dataset_path, batch_shape, data_format, frame_gen_method,
                 is_x_first, is_x_flipped, is_y_flipped, frame_width,
                 num_events_per_frame, chip_size, target_shape=None,
                 label_dict=None):
        self.dataset_path = dataset_path
        self.batch_shape = batch_shape
        self.batch_size = batch_shape[0]
        self.frame_width = frame_width
        self.batch_idx = 0
        self.chip_size = chip_size
        self.target_shape = target_shape
        self.num_events_of_sample = 0
        self.dvs_sample_idx = -1
        self.num_events_per_frame = num_events_per_frame
        self.num_events_per_batch = self.batch_size * num_events_per_frame
        self.x_b_xaddr = self.x_b_yaddr = self.x_b_ts = self.x_b_pol = None
        self.y_b = None
        self.frames_from_sequence = None
        self.event_sequence = None
        self.data_format = data_format
        self.frame_gen_method = frame_gen_method
        self.is_x_first = is_x_first
        self.is_x_flipped = is_x_flipped
        self.is_y_flipped = is_y_flipped

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
        event_sequence = load_event_sequence(filepath, self.chip_size)

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
        if self.num_events_of_sample <= \
                self.num_events_per_batch * (self.batch_idx + 1):
            self.event_sequence = self.next_sequence()

            # Get class label, which is the same for all events in a sequence.
            self.y_b = np.broadcast_to(to_categorical(
                [self.labels[self.dvs_sample_idx]], self.num_classes),
                (self.batch_size, self.num_classes)).astype('float32')

            # Generate frames from events.
            self.frames_from_sequence = get_frames_from_sequence(
                self.event_sequence['x'], self.event_sequence['y'],
                self.event_sequence['pol'], self.num_events_per_frame,
                self.data_format, self.frame_gen_method, self.is_x_first,
                self.is_x_flipped, self.is_y_flipped, self.chip_size,
                self.target_shape)
            # Discard last frames that do not fill a complete batch.
            num_frames = self.batch_size * int(self.num_events_of_sample /
                                               self.num_events_per_batch)
            self.frames_from_sequence = self.frames_from_sequence[:num_frames]

        # From the current event sequence, extract the next bunch of events and
        # stack them as a batch of small sequences.
        self.x_b_xaddr, self.x_b_yaddr, self.x_b_ts, self.x_b_pol = \
            extract_batch(self.event_sequence['x'], self.event_sequence['y'],
                          self.event_sequence['ts'], self.event_sequence['pol'],
                          self.frame_gen_method, self.batch_size,
                          self.batch_idx, self.num_events_per_frame,
                          self.chip_size, self.target_shape)

        self.batch_idx += 1

        return self.x_b_xaddr, self.x_b_yaddr, self.x_b_ts, self.x_b_pol, \
            self.y_b

    def next_eventframe_batch(self):
        return next_eventframe_batch(
            self.x_b_xaddr, self.x_b_yaddr, self.x_b_ts, self.x_b_pol,
            self.is_x_first, self.is_x_flipped, self.is_y_flipped,
            self.batch_shape, self.data_format, self.frame_width)

    def get_frame_batch(self):
        event_idxs = range(self.batch_size * (self.batch_idx - 1),
                           self.batch_size * self.batch_idx)
        return self.frames_from_sequence[event_idxs]


def extract_batch(xaddr, yaddr, timestamps, pol, frame_gen_method, batch_size,
                  batch_idx, num_events_per_frame, chip_size,
                  target_shape=None):
    """Transform a one-dimensional sequence of AER-events into a batch.

    Parameters
    ----------

    frame_gen_method :
    xaddr: ndarray
    yaddr: ndarray
    timestamps: ndarray
    pol: ndarray
    batch_size: int
    batch_idx: int
    num_events_per_frame: int
    chip_size: tuple[int]
    target_shape: Optional[tuple[int]]

    Returns
    -------

    x_b_xaddr, x_b_yaddr, x_b_ts: tuple[ndarray]
        Batch of event sequences.

    """

    from collections import deque

    if target_shape is None:
        target_shape = chip_size
        scale = None
    else:
        scale = [np.true_divide((t - 1), (c - 1)) for t, c in zip(target_shape,
                                                                  chip_size)]

    x_b_xaddr = [deque() for _ in range(batch_size)]
    x_b_yaddr = [deque() for _ in range(batch_size)]
    x_b_ts = [deque() for _ in range(batch_size)]
    x_b_pol = [deque() for _ in range(batch_size)]

    print("Extracting batch of samples à {} events from DVS sequence..."
          "".format(num_events_per_frame))

    for sample_idx in range(batch_size):
        start_event = num_events_per_frame * batch_size * batch_idx + \
                      num_events_per_frame * sample_idx
        event_idxs = range(start_event, start_event + num_events_per_frame)
        event_sums = np.zeros(target_shape, 'int32')
        xaddr_sub = []
        yaddr_sub = []
        for x, y, p in zip(xaddr[event_idxs], yaddr[event_idxs],
                           pol[event_idxs]):
            if scale is not None:
                # Subsample from 240x180 to e.g. 64x64
                x = int(x * scale[0])
                y = int(y * scale[1])
            # Count event at subsampled location. No need to worry about
            # flipping dimensions because the actual frames will be generated
            # someplace else. Here we output only 1d lists.
            add_event_to_frame(event_sums, x, y, p, frame_gen_method)

            xaddr_sub.append(x)
            yaddr_sub.append(y)

        clip_three_sigma(event_sums, frame_gen_method)

        print("Discarded {} events during 3-sigma standardization.".format(
            num_events_per_frame - np.sum(np.abs(event_sums))))

        for x, y, ts, p in zip(xaddr_sub, yaddr_sub, timestamps[event_idxs],
                               pol[event_idxs]):
            if event_sums[x, y] != 0:
                x_b_xaddr[sample_idx].append(x)
                x_b_yaddr[sample_idx].append(y)
                x_b_ts[sample_idx].append(ts)
                x_b_pol[sample_idx].append(p)
                event_sums[x, y] -= np.sign(event_sums[x, y])

    return x_b_xaddr, x_b_yaddr, x_b_ts, x_b_pol


def remove_outliers(timestamps, xaddr, yaddr, pol, x_max=240, y_max=180):
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
    xaddr_valid = np.where(np.array(xaddr) < x_max)
    yaddr_valid = np.where(np.array(yaddr) < y_max)
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
        Chip dimensions, i.e. 1 + largest indices with zero-convention.

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

    return {'x': xaddr, 'y': yaddr, 'ts': timestamps, 'pol': pol}


def get_binary_frame(xaddr, yaddr, timestamps, pol, is_x_first, is_x_flipped,
                     is_y_flipped, shape, data_format, frame_width):
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
    pol: collections.deque
    is_x_first :
    is_x_flipped :
    is_y_flipped :
    shape: tuple
        Include channel dimension even for gray-scale images, e.g. (1, 64, 64)
        if ``data_format='channels_first'``.
    data_format: str
        Either 'channels_first' or 'channels_last'.
    frame_width: int

    Returns
    -------

    binary_frame: ndarray
    """

    # Allocate output array.
    channel_axis = 0 if data_format == 'channels_first' else -1
    binary_frame = np.squeeze(np.zeros(shape), channel_axis)

    # Buffer event sequence because we will be removing elements from original
    # list:
    xaddr_sample = list(xaddr)
    yaddr_sample = list(yaddr)
    ts_sample = list(timestamps)
    pol_sample = list(pol)

    # Need first timestamp of current event sequence to determine when to stop
    # adding events.
    first_ts_of_frame = ts_sample[0] if ts_sample else 0

    # Put events from event sequence buffer into frame, if pixel location is not
    # occupied yet.
    for x, y, ts, p in zip(xaddr_sample, yaddr_sample, ts_sample, pol_sample):
        if binary_frame[y, x] == 0:
            # When adding the event here, we do not care about polarity, because
            # xaddr and yaddr contain only "unit events" that remain after the
            # events are filtered by ``extract_batch``.
            add_event_to_frame(binary_frame, x, y, p, 'rectified_sum',
                               is_x_first, is_x_flipped, is_y_flipped)
            # Can't use .popleft()
            xaddr.remove(x)
            yaddr.remove(y)
            timestamps.remove(ts)
            pol.remove(p)
        if ts - first_ts_of_frame > frame_width:
            # Start next frame if width of frame exceeds time limit.
            break

    return np.expand_dims(binary_frame, channel_axis)


def get_eventframe_sequence(xaddr, yaddr, timestamps, pol, is_x_first,
                            is_x_flipped, is_y_flipped, shape, data_format,
                            frame_width):
    """
    Given a single sequence of x-y-ts events, generate a sequence of binary
    event frames.
    """

    inp = []

    while len(xaddr) > 0:
        inp.append(get_binary_frame(
            xaddr, yaddr, timestamps, pol, is_x_first, is_x_flipped,
            is_y_flipped, shape, data_format, frame_width))

    return np.stack(inp, -1)


def next_eventframe_batch(x_b_xaddr, x_b_yaddr, x_b_ts, x_b_pol, is_x_first,
                          is_x_flipped, is_y_flipped, shape, data_format,
                          frame_width):
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
            x_b_pol[sample_idx], is_x_first, is_x_flipped, is_y_flipped,
            shape[1:], data_format, frame_width)

    return input_b_l


def get_frames_from_sequence(xaddr, yaddr, pol, num_events_per_frame,
                             data_format, frame_gen_method, is_x_first,
                             is_x_flipped, is_y_flipped, chip_size,
                             target_shape=None):
    """
    Extract ``num_events_per_frame`` events from a one-dimensional sequence of
    AER-events. The events are spatially subsampled to ``target_shape``, and
    standardized to [0, 1] using 3-sigma normalization. The resulting events
    are binned into a frame. The function operates on the events in
    ``xaddr`` etc sequentially until all are processed into frames.
    """

    if target_shape is None:
        target_shape = chip_size
        scale = None
    else:
        scale = [np.true_divide((t - 1), (c - 1)) for t, c in zip(target_shape,
                                                                  chip_size)]

    num_frames = int(len(xaddr) / num_events_per_frame)
    frames = np.zeros([num_frames] + list(target_shape), 'float32')

    print("Extracting {} frames from DVS event sequence.".format(num_frames))

    # Iterate for as long as there are events in the sequence.
    for sample_idx in range(num_frames):
        sample = frames[sample_idx]
        event_idxs = range(num_events_per_frame * sample_idx,
                           num_events_per_frame * (sample_idx + 1))

        # Loop over ``num_events_per_frame`` events
        for x, y, p in zip(xaddr[event_idxs], yaddr[event_idxs],
                           pol[event_idxs]):
            if scale is not None:
                # Subsample from 240x180 to e.g. 64x64
                x = int(x * scale[0])
                y = int(y * scale[1])

            add_event_to_frame(sample, x, y, p, frame_gen_method, is_x_first,
                               is_x_flipped, is_y_flipped)

        clip_three_sigma(sample, frame_gen_method)

    scale_event_frames(frames, frame_gen_method)

    channel_axis = 1 if data_format == 'channels_first' else -1

    return np.expand_dims(frames, channel_axis)


def add_event_to_frame(frame, x, y, p, frame_gen_method='rectified_sum',
                       is_x_first=True, is_x_flipped=False, is_y_flipped=False):

    x_max, y_max = frame.shape

    x = x_max - 1 - x if is_x_flipped else x
    y = y_max - 1 - y if is_y_flipped else y

    idx0, idx1 = (x, y) if is_x_first else (y, x)

    incr = 1
    sign = 1
    if frame_gen_method == 'signed_sum':
        sign = 1 if p else -1

    frame[idx0, idx1] += sign * incr


def clip_three_sigma(frame, frame_gen_method):
    # Compute standard deviation of event-sum distribution after removing
    # zeros, then clip number of events per pixel to three-sigma.
    if frame_gen_method == 'rectified_sum':
        sigma = np.std(frame[np.nonzero(frame)])
        a_min = 0
        a_max = 3 * sigma
        # It would make more sense to use the same a_min, a_max as for
        # 'signed_sum', but we don't because jAER implements it like this.
    elif frame_gen_method == 'signed_sum':
        sigma = np.std(frame)
        mean = np.mean(frame)
        a_min = mean - 1.5 * sigma
        a_max = mean + 1.5 * sigma
    else:
        a_min = np.min(frame)
        a_max = np.max(frame)

    np.clip(frame, a_min, a_max, frame)


def scale_event_frames(frames, frame_gen_method):
    for frame in frames:
        a_min = np.min(frame)
        a_max = np.max(frame)
        np.true_divide(frame - a_min, a_max - a_min, frame)
