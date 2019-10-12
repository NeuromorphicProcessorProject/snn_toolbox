# -*- coding: utf-8 -*-

"""
The purpose of this module is to provide an executable for running the SNN
conversion toolbox, either from terminal or using a GUI.

During installation of the toolbox, python creates an entry point to the `main`
function of this module. See :ref:`running` for how call this executable.

@author: rbodo
"""

import argparse
import os


def main(filepath=None):
    """Entry point for running the toolbox.

    Note
    ----

    There is no need to call this function directly, because python sets up an
    executable during :ref:`installation` that can be called from terminal.

    """
    from snntoolbox.bin.utils import update_setup, run_pipeline

    if filepath is not None:
        config = update_setup(filepath)
        run_pipeline(config)
        return

    parser = argparse.ArgumentParser(
        description='Run SNN toolbox to convert an analog neural network into '
                    'a spiking neural network, and optionally simulate it.')
    parser.add_argument('config_filepath', nargs='?',
                        help='Path to configuration file.')
    parser.add_argument('-t', '--terminal', action='store_true',
                        help='Set this flag to run the toolbox from terminal. '
                             'Omit this flag to open GUI.')
    args = parser.parse_args()

    _filepath = os.path.abspath(args.config_filepath)
    if _filepath is not None:
        config = update_setup(_filepath)

        if args.terminal:
            run_pipeline(config)
        else:
            from snntoolbox.bin.gui import gui
            gui.main()
    else:
        if args.terminal:
            parser.error("When using the SNN toolbox from terminal, a "
                         "config_filepath argument must be provided.")
            return
        else:
            from snntoolbox.bin.gui import gui
            gui.main()


if __name__ == '__main__':
    main()
