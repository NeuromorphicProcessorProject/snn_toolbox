# -*- coding: utf-8 -*-

"""Script to run SNN toolbox either from console or as GUI."""

import argparse
import os


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

    if filepath is not None:
        assert os.path.isfile(filepath), \
            "Configuration file not found at {}.".format(filepath)
        from bin.utils import update_setup
        config = update_setup(filepath)

        if args.terminal:
            from bin.utils import test_full
            test_full(config)
        else:
            from bin.gui import gui
            gui.main()
    else:
        if args.terminal:
            parser.error("When using the SNN toolbox from terminal, a "
                         "config_filepath argument must be provided.")
            return
        else:
            from bin.gui import gui
            gui.main()

if __name__ == '__main__':
    main()
