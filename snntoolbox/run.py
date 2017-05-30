# -*- coding: utf-8 -*-

"""Script to run SNN toolbox either from console or as GUI."""

import argparse
from snntoolbox.config import update_setup


def main():
    """Entry point for running the toolbox either from terminal or as GUI."""
    parser = argparse.ArgumentParser(
        description='Run SNN toolbox to convert an analog neural network into '
                    'a spiking neural network, and optionally simulate it.')
    parser.add_argument('settings_filepath', nargs='?',
                        help='Path to text file containing the settings dict.')
    parser.add_argument('-t', '--terminal', action='store_true',
                        help='Set this flag to run the toolbox from terminal. '
                             'Omit this flag to open GUI.')
    args = parser.parse_args()

    if args.terminal:
        if args.settings_filepath is None:
            parser.error("When using the SNN toolbox from terminal, a "
                         "settings_filepath argument must be provided.")
            return
        with open(args.settings_filepath, 'r') as f:
            update_setup(eval(f.read()))
        from snntoolbox.core.pipeline import test_full
        test_full()
    else:
        if args.settings_filepath is not None:
            with open(args.settings_filepath, 'r') as f:
                update_setup(eval(f.read()))
        from snntoolbox.gui import gui
        gui.main()

if __name__ == '__main__':
    main()
