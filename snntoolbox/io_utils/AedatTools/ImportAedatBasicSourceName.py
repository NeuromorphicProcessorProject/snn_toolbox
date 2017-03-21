# -*- coding: utf-8 -*-

"""Import basic source name"""


def import_aedat_basic_source_name(inp):
    """

    Parameters
    ----------
    inp:
        A device name.

    Returns
    -------
    :
        The key device name associated with ``input``.

    """

    devices = {
        'file': 'File',
        'network': 'Network',
        'dvs128': 'Dvs128',
        'tmpdiff128': 'Dvs128',
        'davis240a': 'Davis240A',
        'sbret10': 'Davis240A',
        'davis240b': 'Davis240B',
        'sbret20': 'Davis240B',
        'davis240c': 'Davis240C',
        'sbret21': 'Davis240C',
        'davis128mono': 'Davis128Mono',
        'davis128rgb': 'Davis128Rgb',
        'davis128': 'Davis128Rgb',
        'davis208mono': 'Davis208Mono',
        'davis208rgbw': 'Davis208Mono',
        'pixelparade': 'Davis208Rgbw',
        'sensdavis192': 'Davis208Rgbw',
        'davis208': 'Davis208Rgbw',
        'davis346amono': 'Davis346AMono',
        'davis346argb': 'Davis346ARgb',
        'davis346': 'Davis346ARgb',
        'davis346bmono': 'Davis346BMono',
        'davis346brgb': 'Davis346BRgb',
        'davis346b': 'Davis346BRgb',
        'davis346cbsi': 'Davis346CBsi',
        'davis346bsi': 'Davis346CBsi',
        'davis640mono': 'Davis640Mono',
        'davis640rgb': 'Davis640Rgb',
        'davis640': 'Davis640Rgb',
        'davishet640mono': 'DavisHet640Mono',
        'davishet640rgbw': 'DavisHet640Rgbw',
        'cdavis640': 'DavisHet640Rgbw',
        'cdavis640rgbw': 'DavisHet640Rgbw',
        'das1': 'Das1',
        'cochleaams1c': 'CochleaAms1c'}

    return devices.get(inp.lower()[:-2], 'Dvs128')  # Cut off '\r'
