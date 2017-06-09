# -*- coding: utf-8 -*-

"""Script to run SNN toolbox either from console or as GUI."""

import os
import argparse


def main():
    """Entry point for running the toolbox either from terminal or as GUI."""
    parser = argparse.ArgumentParser(
        description='Run SNN toolbox to convert an analog neural network into '
                    'a spiking neural network, and optionally simulate it.')
    parser.add_argument('config_filepath', nargs='?',
                        help='Path to configuration file.')
    parser.add_argument('-t', '--terminal', action='store_true',
                        help='Set this flag to run the toolbox from terminal. '
                             'Omit this flag to open GUI.')
    args = parser.parse_args()

    filepath = os.path.abspath(args.config_filepath)
#    filepath = '/mnt/2646BAF446BAC3B9/Repositories/NPP/snn_toolbox/examples/models/binarynet/config'
 #   args.terminal = True
    if filepath is not None:
        assert os.path.isfile(filepath), \
            "Configuration file not found at {}.".format(filepath)
        from snntoolbox.config import update_setup
        config = update_setup(filepath)

        if args.terminal:
            from snntoolbox.core.pipeline import test_full
            test_full(config)
        else:
            from snntoolbox.gui import gui
            gui.main()
    else:
        if args.terminal:
            parser.error("When using the SNN toolbox from terminal, a "
                         "config_filepath argument must be provided.")
            return
        else:
            from snntoolbox.gui import gui
            gui.main()

if __name__ == '__main__':
    main()
