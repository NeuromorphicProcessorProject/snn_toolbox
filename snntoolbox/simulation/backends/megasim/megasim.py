# -*- coding: utf-8 -*-
""" MegaSim spiking neuron simulator.

A collection of helper functions used to get MegaSim's path and executable.

the configuration file will be stored at
$HOME/.snntoolbox/preferences/megasim_config.json

Assumes that have write access to the home folder.

@author: evan
"""

from __future__ import print_function, unicode_literals
from __future__ import division, absolute_import
from future import standard_library

import os
import sys
import json

standard_library.install_aliases()


def megasim_path():
    """

    Returns
    -------

    """

    # first check if the .snntoolbox folder exists
    home_path = os.environ["HOME"]
    snntoobox_path_root = home_path+"/.snntoolbox/"
    snntoobox_preferences_path = snntoobox_path_root+"preferences/"
    megasim_config_json_fname = "megasim_config.json"
    if os.path.isdir(snntoobox_path_root):
        # config folder found, check if the preferences folder is there
        if os.path.isdir(snntoobox_preferences_path):
            try:
                megasim_file_config = open(snntoobox_preferences_path +
                                           megasim_config_json_fname, "r")
                megaconfig = json.load(megasim_file_config)
                megasim_path_is = megaconfig["MegaSim_path"]
                print("MegaSim folder is "+megasim_path_is)
            except FileNotFoundError:
                # megasim config json file not found, ask the user for megasims
                # path and check if the executable is there
                print("MegaSim's config file not found.")
                new_path_is = input("Please enter the full path to megasim"
                                    " executable: ")
                if new_path_is[-1] != "/":
                    new_path_is = new_path_is+"/"

                print("Checking if MegaSim executable exists at "+new_path_is)
                if not os.path.isfile(new_path_is+"megasim"):
                    print("MegaSim executable not found in "+new_path_is)
                    sys.exit(1)
                print("Creating the MegaSim config file")
                f = open(snntoobox_preferences_path+megasim_config_json_fname,
                         "w")
                build_line = '{"MegaSim_path": "'+new_path_is+'"}'
                f.write(build_line)
                f.write("\n")
                f.close()

                megasim_path_is = new_path_is
        else:
            print("snntoolbox preferences directory not found")
            megasim_path_is = " "

    else:
        print("snntoolbox config directory not found")
        cur_dir = os.getcwd()
        os.chdir(home_path)
        os.makedirs(snntoobox_preferences_path)
        print("MegaSim's config file not found.")
        new_path_is = input("Please enter the full path to megasim executable:"
                            " ")
        if new_path_is[-1] != "/":
            new_path_is = new_path_is + "/"

        print("Checking if MegaSim executable exists at " + new_path_is)
        if not os.path.isfile(new_path_is + "megasim"):
            print("MegaSim executable not found in " + new_path_is)
            sys.exit(1)
        print("Creating the MegaSim config file")
        f = open(snntoobox_preferences_path + megasim_config_json_fname, "w")
        build_line = '{"MegaSim_path": "' + new_path_is + '"}'
        f.write(build_line)
        f.write("\n")
        f.close()

        # megasim_path_is = new_path_is
        os.chdir(cur_dir)
        megasim_path_is = " "

    return megasim_path_is
