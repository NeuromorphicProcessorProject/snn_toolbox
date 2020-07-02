# -*- coding: utf-8 -*-

"""
Tools to load DVS sequence, preprocess it, and create batches of event-frames
for use in a time-stepped simulator.
"""

import os
import numpy as np
from collections import deque
from more_itertools import unique_everseen
from snntoolbox.datasets.utils import to_categorical


class DVSIterator(object):
    def __init__(self, dataset_path, batch_shape, data_format,
                 frame_gen_method, is_x_first, is_x_flipped, is_y_flipped,
                 frame_width, num_events_per_frame, maxpool_subsampling,
                 do_clip_three_sigma, chip_size, target_shape=None,
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
        self.y_b = None
        self.frames_from_sequence = None
        self.event_sequence = None
        self.event_deques_batch = None
        self.data_format = data_format
        self.frame_gen_method = frame_gen_method
        self.is_x_first = is_x_first
        self.is_x_flipped = is_x_flipped
        self.is_y_flipped = is_y_flipped
        self.maxpool_subsampling = maxpool_subsampling
        self.do_clip_three_sigma = do_clip_three_sigma

        # Count the number of samples and classes
        classes = [subdir for subdir in sorted(os.listdir(dataset_path))
                   if os.path.isdir(os.path.join(dataset_path, subdir))]

        self.label_dict = dict(zip(classes, range(len(classes)))) \
            if not label_dict else label_dict
        self.num_classes = len(self.label_dict)
        assert self.num_classes == len(classes), \
            "The number of classes provided by label_dict {} does not match " \
            "the number of subdirectories found in dataset_path {}.".format(
                self.label_dict, self.dataset_path)

        self.filenames = []
        labels = []
        self.num_samples = 0
        for subdir in classes:
            for fname in sorted(os.listdir(os.path.join(dataset_path,
                                                        subdir))):
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
        event_list = load_event_list(filepath, self.chip_size)

        # Update statistics of current sequence.
        self.num_events_of_sample = len(event_list)
        print("Total number of events of this sample: {}.".format(
            self.num_events_of_sample))
        print("Number of batches: {}.".format(
            self.num_events_of_sample // self.num_events_per_batch + 1))

        # Reset batch index, because new sequence will be used to generate new
        # batches.
        self.batch_idx = 0

        return event_list

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
                self.event_sequence, self.num_events_per_frame,
                self.data_format, self.frame_gen_method, self.is_x_first,
                self.is_x_flipped, self.is_y_flipped, self.maxpool_subsampling,
                self.do_clip_three_sigma, self.chip_size, self.target_shape)
            # Discard last frames that do not fill a complete batch.
            num_frames = self.batch_size * (self.num_events_of_sample //
                                            self.num_events_per_batch + 1)
            self.frames_from_sequence = self.frames_from_sequence[:num_frames]

        # From the current event sequence, extract the next bunch of events and
        # stack them as a batch of small sequences.
        self.event_deques_batch = extract_batch(
            self.event_sequence, self.frame_gen_method, self.batch_size,
            self.batch_idx, self.num_events_per_frame,
            self.maxpool_subsampling, self.do_clip_three_sigma, self.chip_size,
            self.target_shape)

        self.batch_idx += 1

        return self.event_deques_batch, self.y_b

    def next_eventframe_batch(self):
        return next_eventframe_batch(self.event_deques_batch, self.is_x_first,
                                     self.is_x_flipped, self.is_y_flipped,
                                     self.batch_shape, self.data_format,
                                     self.frame_width, self.frame_gen_method)

    def get_frame_batch(self):
        event_idxs = range(self.batch_size * (self.batch_idx - 1),
                           self.batch_size * self.batch_idx)
        return self.frames_from_sequence[event_idxs]

    def remaining_events_of_current_batch(self):
        num_events = 0
        for d in self.event_deques_batch:
            num_events += len(d)
        return num_events


def extract_batch(event_list, frame_gen_method, batch_size,
                  batch_idx, num_events_per_frame, maxpool_subsampling,
                  do_clip_three_sigma, chip_size, target_shape=None):
    """Transform a one-dimensional sequence of AER-events into a batch.

    Parameters
    ----------

    event_list: list[tuple]
        [(x1, y1, t1, p1), (x2, y2, t2, p2), ...]
    frame_gen_method: str
    batch_size: int
    batch_idx: int
    num_events_per_frame: int
    chip_size: tuple[int]
    target_shape: Optional[tuple[int]]
    maxpool_subsampling: bool
    do_clip_three_sigma: bool

    Returns
    -------

    event_deques_list: list[deque]
        List of length ``batch_size``, with an event deque for each sample in
        the batch. The deques contain event tuples (x, y, t, p).

    """

    if target_shape is None:
        target_shape = chip_size
        scale = None
    else:
        scale = [np.true_divide((t - 1), (c - 1)) for t, c in zip(target_shape,
                                                                  chip_size)]

    num_channels = 2 if has_polarity_channels(frame_gen_method) else 1

    event_deques_list = [deque() for _ in range(batch_size)]

    print("Extracting batch of samples à {} events from DVS sequence..."
          "".format(num_events_per_frame))

    for sample_idx in range(batch_size):
        start_event = num_events_per_frame * batch_size * batch_idx + \
                      num_events_per_frame * sample_idx
        event_idxs = slice(start_event, start_event + num_events_per_frame)
        event_sums = np.zeros(list(target_shape) + [num_channels], 'int32')
        frame_event_list = []
        for x, y, t, p in event_list[event_idxs]:
            if scale is not None:
                # Subsample from 240x180 to e.g. 64x64
                x = int(x * scale[0])
                y = int(y * scale[1])

            # Need to remove polarity here if frame_gen_method ==
            # 'rectified_sum', so that we can discard an otherwise identical
            # event with opposite polarity during maxpool_subsampling.
            pp = 1 if frame_gen_method == 'rectified_sum' else p
            frame_event_list.append((x, y, t, pp))

        num_events = len(frame_event_list)

        if maxpool_subsampling:
            frame_event_list = list(unique_everseen(frame_event_list))
            num_events_after_subsampling = len(frame_event_list)
            print("Discarded {} events during subsampling.".format(
                num_events - num_events_after_subsampling))
        else:
            num_events_after_subsampling = num_events

        if do_clip_three_sigma:
            for x, y, t, p in frame_event_list:
                # Count event at subsampled location. No need to worry about
                # flipping dimensions because the actual frames will be
                # generated someplace else. Here we output only 1d lists.
                add_event_to_frame(event_sums, x, y, p, frame_gen_method)

            event_sums = clip_three_sigma(event_sums, frame_gen_method)

            print("Discarded {} events during 3-sigma standardization.".format(
                num_events_after_subsampling - np.sum(np.abs(event_sums))))

        for x, y, t, p in frame_event_list:
            pp = p if has_polarity_channels(frame_gen_method) else 0
            if not do_clip_three_sigma or event_sums[x, y, pp] != 0:
                event_deques_list[sample_idx].append((x, y, t, p))
                event_sums[x, y, pp] -= np.sign(event_sums[x, y, pp])

    return event_deques_list


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


def load_event_list(filename, xyrange=None):
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

    dvs_sequence: list[tuple]
        List of tuples, where each tuple consists of the following items:

            - x: int
                The x-addresses.
            - y: int
                The y-addresses.
            - t: int
                The timestamps.
            - p: int
                The polarity.

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

    return [(x, y, t, p) for x, y, t, p in zip(xaddr, yaddr, timestamps, pol)]


def get_binary_frame(event_deque, is_x_first, is_x_flipped, is_y_flipped,
                     shape, data_format, frame_width, frame_gen_method):
    """
    Put events from event sequence into a shallow frame of at most one event
    per pixel. Stop if the time between the current and the oldest event
    exceeds ``frame_width``. Note that the events that have been added to the
    binary frame are removed from the input sequence!

    Parameters
    ----------

    event_deque: collections.deque
    is_x_first :
    is_x_flipped :
    is_y_flipped :
    shape: tuple
        Include channel dimension even for gray-scale images, e.g. (1, 64, 64)
        if ``data_format='channels_first'``.
    data_format: str
        Either 'channels_first' or 'channels_last'.
    frame_width: int
    frame_gen_method: str

    Returns
    -------

    binary_frame: ndarray
    """

    # Allocate output array.
    is_channels_first = data_format == 'channels_first'
    if is_channels_first:
        num_channels, x_max, y_max = shape
    else:
        x_max, y_max, num_channels = shape
    binary_frame = np.zeros((x_max, y_max, num_channels))

    # Buffer event sequence because we will be removing elements from original
    # list:
    event_list = list(event_deque)

    # Need first timestamp of current event sequence to determine when to stop
    # adding events.
    first_ts_of_frame = event_list[0][2] if event_list else 0

    # Put events from event sequence buffer into frame, if pixel location is
    # not occupied yet.
    for x, y, t, p in event_list:
        x_flipped = x_max - 1 - x if is_x_flipped else x
        y_flipped = y_max - 1 - y if is_y_flipped else y

        idx0, idx1 = (x_flipped, y_flipped) if is_x_first else (y_flipped,
                                                                x_flipped)
        pp = p if num_channels > 1 else 0
        if binary_frame[idx0, idx1, pp] == 0:
            spike = 1
            if p == 0 and frame_gen_method in ['signed_polarity_channels',
                                               'signed_sum']:
                spike = -1
            binary_frame[idx0, idx1, pp] = spike
            event_deque.remove((x, y, t, p))
        if t - first_ts_of_frame > frame_width:
            # Start next frame if width of frame exceeds time limit.
            break

    if is_channels_first:
        binary_frame = np.moveaxis(binary_frame, -1, 1)

    return binary_frame


def get_eventframe_sequence(event_deque, is_x_first, is_x_flipped,
                            is_y_flipped, shape, data_format, frame_width,
                            frame_gen_method):
    """
    Given a single sequence of x-y-ts events, generate a sequence of binary
    event frames.
    """

    inp = []

    while len(event_deque) > 0:
        inp.append(get_binary_frame(event_deque, is_x_first, is_x_flipped,
                                    is_y_flipped, shape, data_format,
                                    frame_width, frame_gen_method))

    return np.stack(inp, -1)


def next_eventframe_batch(event_deques_batch, is_x_first, is_x_flipped,
                          is_y_flipped, shape, data_format, frame_width,
                          frame_gen_method):
    """
    Given a batch of x-y-ts event sequences, generate a batch of binary event
    frames that can be used in a time-stepped simulator.
    """

    # Allocate output array.
    input_b_l = np.zeros(shape, 'float32')

    # Generate each frame in batch sequentially.
    for sample_idx in range(shape[0]):
        input_b_l[sample_idx] = get_binary_frame(
            event_deques_batch[sample_idx], is_x_first, is_x_flipped,
            is_y_flipped, shape[1:], data_format, frame_width,
            frame_gen_method)

    return input_b_l


def get_frames_from_sequence(event_list, num_events_per_frame, data_format,
                             frame_gen_method, is_x_first, is_x_flipped,
                             is_y_flipped, maxpool_subsampling,
                             do_clip_three_sigma, chip_size,
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
    num_channels = 2 if has_polarity_channels(frame_gen_method) else 1
    num_frames = len(event_list) // num_events_per_frame + 1
    frames = np.zeros([num_frames] + list(target_shape) + [num_channels],
                      'float32')

    print("Extracting {} frames from DVS event sequence.".format(num_frames))

    # Iterate for as long as there are events in the sequence.
    for sample_idx in range(num_frames):
        sample = frames[sample_idx]
        event_idxs = slice(num_events_per_frame * sample_idx,
                           num_events_per_frame * (sample_idx + 1))

        # Loop over ``num_events_per_frame`` events
        frame_event_list = []
        for x, y, t, p in event_list[event_idxs]:
            if scale is not None:
                # Subsample from 240x180 to e.g. 64x64
                x = int(x * scale[0])
                y = int(y * scale[1])

            pp = 1 if frame_gen_method == 'rectified_sum' else p
            frame_event_list.append((x, y, t, pp))

        if maxpool_subsampling:
            frame_event_list = list(unique_everseen(frame_event_list))

        for x, y, t, p in frame_event_list:
            add_event_to_frame(sample, x, y, p, frame_gen_method, is_x_first,
                               is_x_flipped, is_y_flipped)

        # sample = scale_event_frames(sample, frame_gen_method)
        if do_clip_three_sigma:
            frames[sample_idx] = clip_three_sigma(sample, frame_gen_method)
        else:
            frames[sample_idx] = sample

    frames = scale_event_frames(frames)

    if data_format == 'channels_first':
        frames = np.moveaxis(frames, -1, 1)
    return frames


def add_event_to_frame(frame, x, y, p, frame_gen_method='rectified_sum',
                       is_x_first=True, is_x_flipped=False,
                       is_y_flipped=False):

    x_max, y_max, _ = frame.shape

    x = x_max - 1 - x if is_x_flipped else x
    y = y_max - 1 - y if is_y_flipped else y

    idx0, idx1 = (x, y) if is_x_first else (y, x)

    if frame_gen_method == 'signed_sum':
        frame[idx0, idx1] += 1 if p else -1
    elif frame_gen_method == 'rectified_sum':
        frame[idx0, idx1] += 1
    elif frame_gen_method == 'rectified_polarity_channels':
        frame[idx0, idx1, p] += 1
    elif frame_gen_method == 'signed_polarity_channels':
        frame[idx0, idx1, p] += 1 if p else -1


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
        frame_nz = frame[np.nonzero(frame)]
        sigma = np.std(frame_nz)
        mean = np.mean(frame_nz)
        a_min = mean - 1.5 * sigma
        a_max = mean + 1.5 * sigma
    elif frame_gen_method == 'rectified_polarity_channels':
        frame_off = frame[:, :, 0]
        frame_on = frame[:, :, 1]
        sigma_off = np.std(frame_off[np.nonzero(frame_off)])
        sigma_on = np.std(frame_on[np.nonzero(frame_on)])
        a_min = 0
        a_max = [[[3 * sigma_off, 3 * sigma_on]]]  # Assumes channels_last.
    elif frame_gen_method == 'signed_polarity_channels':
        frame_off = frame[:, :, 0]
        frame_on = frame[:, :, 1]
        sigma_off = np.std(frame_off[np.nonzero(frame_off)])
        sigma_on = np.std(frame_on[np.nonzero(frame_on)])
        a_min = [[[-3 * sigma_off, 0]]]
        a_max = [[[0, 3 * sigma_on]]]
    else:
        a_min = np.min(frame)
        a_max = np.max(frame)

    np.clip(frame, a_min, a_max, frame)

    return np.round(frame)


def scale_event_frames(frames):
    # Do not scale frames individually, as this would increase the power
    # contained in flatly-distributed images. The pure DVS data is never scaled
    # and would not be able to recover this difference in distribution.
    a_min = np.min(frames)
    a_max = np.max(frames)
    div = 1 if a_max == 0 else a_max
    np.true_divide(frames - a_min, div, frames)

    # for frame in frames:
    #     a_min = np.min(frame)
    #     a_max = np.max(frame)
    #     div = 1 if a_min == a_max else a_max - a_min
    #     np.true_divide(frame - a_min, div, frame)

    return frames


def has_polarity_channels(frame_gen_method):
    return 'polarity_channels' in frame_gen_method
