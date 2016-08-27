"""Running Collection of Max-Pooling Experiments.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

from __future__ import print_function
import os
import argparse
import json

import pooling_exps as pe

max_pool_type = ["avg_max", "fir_max", "exp_max"]
online_normalize = "false"
normalize = "true"
percentile_list = [99.9]

home_path = os.environ["HOME"]
config_path = os.path.join(home_path, ".snntoolbox")
pref_dir = os.path.join(config_path, "preferences")


def maxpool_collection(models):
    """Running Max-Pooling collection.

    Parameters
    ----------
    models : string
        The destination of the model collections definition.
    """
    json_file = open(os.path.join(pref_dir, models), "r")
    model_dict = json.load(json_file)

    for model_name in model_dict:
        for percentile in percentile_list:
            for pool_type in max_pool_type:
                exp_name = pool_type.replace("_", "-")+"-" + \
                    model_dict[model_name][0]+"-" + \
                    model_dict[model_name][1]+"-"+str(percentile)
                if online_normalize == "false":
                    exp_name += "-no-online-normalization"

                pe.maxpool_exp(exp_name, model_name,
                               model_dict[model_name][0]+".json",
                               model_dict[model_name][0], normalize,
                               online_normalize,
                               pool_type, percentile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running Max-Pooling \
                                     Experiments Collection by Yuhuang Hu")
    parser.add_argument("-m", "--models", type=str,
                        help="Destination of model collections.")
    args = parser.parse_args()
    maxpool_collection(**vars(args))
