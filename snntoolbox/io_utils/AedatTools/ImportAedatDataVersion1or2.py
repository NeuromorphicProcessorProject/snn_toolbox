# -*- coding: utf-8 -*-

"""
Import aedat version 1 or 2.
"""

import numpy as np


def import_aedat_dataversion1or2(info):
    """

    Parameters
    ----------
    info :
    """

    # The formatVersion dictates whether there are 6 or 8 bytes per event.
    if info['formatVersion'] == 1:
        num_bytes_per_event = 6
        addr_precision = np.dtype([('addr', '>u2'), ('ts', '>u4')])
    else:
        num_bytes_per_event = 8
        addr_precision = np.dtype([('addr', '>u4'), ('ts', '>u4')])

    file_handle = info['fileHandle']

    # Find the number of events, assuming that the file position is just at the
    # end of the headers.
    file_handle.seek(0, 2)
    num_events_in_file = int(np.floor(
        (file_handle.tell() - info['beginningOfDataPointer']) /
        num_bytes_per_event))
    info['numEventsInFile'] = num_events_in_file

    # Check the startEvent and endEvent parameters
    if 'startEvent' not in info:
        info['startEvent'] = 0
    assert info['startEvent'] <= num_events_in_file
    if 'endEvent' not in info:
        info['endEvent'] = num_events_in_file
    if 'startPacket' in info:
        print("The startPacket parameter is set, but range by packets is not "
              "available for .aedat version < 3 files")
    if 'endPacket' in info:
        print("The endPacket parameter is set, but range by events is not "
              "available for .aedat version < 3 files")
    if info['endEvent'] > num_events_in_file:
        print("The file contains {}; the endEvent parameter is {}; reducing "
              "the endEvent parameter accordingly.".format(num_events_in_file,
                                                           info['endEvents']))
        info['endEvent'] = num_events_in_file
    assert info['startEvent'] < info['endEvent']

    num_events_to_read = int(info['endEvent'] - info['startEvent'])

    # Read events
    file_handle.seek(info['beginningOfDataPointer'] + num_bytes_per_event *
                     info['startEvent'])
    all_events = np.fromfile(file_handle, addr_precision, num_events_to_read)

    all_addr = np.array(all_events['addr'])
    all_ts = np.array(all_events['ts'])

    # Trim events outside time window.
    # This is an inefficent implementation, which allows for non-monotonic
    # timestamps.

    if 'startTime' in info:
        temp_index = np.nonzero(all_ts >= info['startTime'] * 1e6)
        all_addr = all_addr[temp_index]
        all_ts = all_ts[temp_index]

    if 'endTime' in info:
        temp_index = np.nonzero(all_ts <= info['endTime'] * 1e6)
        all_addr = all_addr[temp_index]
        all_ts = all_ts[temp_index]

    # DAVIS. In the 32-bit address:
    # bit 32 (1-based) being 1 indicates an APS sample
    # bit 11 (1-based) being 1 indicates a special event
    # bits 11 and 32 (1-based) both being zero signals a polarity event
    aps_or_imu_mask = int('80000000', 16)
    aps_or_imu_logical = np.bitwise_and(all_addr, aps_or_imu_mask)
    signal_or_special_mask = int('400', 16)
    signal_or_special_logical = np.bitwise_and(all_addr, signal_or_special_mask)
    polarity_logical = np.logical_and(np.logical_not(aps_or_imu_logical),
                                      np.logical_not(signal_or_special_logical))

    # These masks are used for both frames and polarity events, so are defined
    # outside of the following if statement
    y_mask = int('7FC00000', 16)
    y_shift_bits = 22
    x_mask = int('003FF000', 16)
    x_shift_bits = 12

    output = {'data': {}}

    # Polarity(DVS) events
    if ('dataTypes' not in info or 'polarity' in info['dataTypes']) \
            and any(polarity_logical):
        output['data']['polarity'] = {}
        output['data']['polarity']['timeStamp'] = all_ts[polarity_logical]
        # Y addresses
        output['data']['polarity']['y'] = np.array(np.right_shift(
            np.bitwise_and(all_addr[polarity_logical], y_mask), y_shift_bits),
            'int32')
        # X addresses
        output['data']['polarity']['x'] = np.array(np.right_shift(
            np.bitwise_and(all_addr[polarity_logical], x_mask), x_shift_bits),
            'int32')
        # Polarity bit
        output['data']['polarity']['polarity'] = np.array(np.equal(
            np.right_shift(all_addr[polarity_logical], 11) % 2, 1), 'int32')

    output['info'] = info

    # calculate numEvents fields; also find first and last timeStamps
    output['info']['firstTimeStamp'] = np.infty
    output['info']['lastTimeStamp'] = 0

    if 'polarity' in output['data']:
        output['data']['polarity']['numEvents'] = \
            len(output['data']['polarity']['timeStamp'])
        # noinspection PyTypeChecker
        if output['data']['polarity']['timeStamp'][0] < \
                output['info']['firstTimeStamp']:
            # noinspection PyTypeChecker
            output['info']['firstTimeStamp'] = \
                output['data']['polarity']['timeStamp'][0]
        # noinspection PyTypeChecker
        if output['data']['polarity']['timeStamp'][-1] > \
                output['info']['lastTimeStamp']:
            # noinspection PyTypeChecker
            output['info']['lastTimeStamp'] = \
                output['data']['polarity']['timeStamp'][-1]

    return output
