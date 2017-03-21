# -*- coding: utf-8 -*-

"""
Import aedat file.
"""

from snntoolbox.io_utils.AedatTools.ImportAedatHeaders import \
    import_aedat_headers
from snntoolbox.io_utils.AedatTools.ImportAedatDataVersion1or2 import \
    import_aedat_dataversion1or2


def import_aedat(args):
    """

    Parameters
    ----------
    args :

    Returns
    -------

    """

    output = {'info': args}

    with open(output['info']['filePathAndName'], 'rb') as \
            output['info']['fileHandle']:
        output['info'] = import_aedat_headers(output['info'])
        return import_aedat_dataversion1or2(output['info'])
