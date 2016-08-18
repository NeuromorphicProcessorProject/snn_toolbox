"""Running experiments for max-pooling experiments.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

from __future__ import print_function
import os
import argparse
import json

import snntoolbox

home_path = os.environ["HOME"]
config_path = os.path.join(home_path, ".snntoolbox")
pref_dir = os.path.join(config_path, "preferences")
log_dir = os.path.join(home_path, "workspace", "snntoolbox-log", "pool-exps")
data_dir = os.path.join(config_path, "datasets")


def maxpool_exp(exp_name, model_name, pref_name, dataset,
                normalize, pool_type):
    """Max-Pooling experiment routine.

    Parameters
    ----------
    exp_name : string
        the name of the experiment
    model_name : string
        the name of the model
    pref_name : string
        the name of the perference
    dataset : string
        the name of the dataset, mnist or cifar10
    normalizing : string
        true : perform normalization and evaluation
        false: otherwise
    pool_type : string
        the name of the max pooling type
        "avg_max" or "fir_max"
    """
    pref_path = os.path.join(pref_dir, pref_name)
    log_path = os.path.join(log_dir, exp_name)
    data_path = os.path.join(data_dir, dataset)

    if not os.path.exists(pref_path):
        raise ValueError("[MESSAGE] The target preference "
                         "file %s is not existed!" % (pref_path))

    if not os.path.isdir(log_path):
        os.mkdir(log_path)

    print ("[MESSAGE] Running experiment %s." % (exp_name))
    print ("[MESSAGE] Loading Experiment settings.")
    settings = json.load(open(pref_path))

    settings["log_dir_of_current_run"] = log_path
    settings["runlabel"] = exp_name
    settings["dataset_path"] = data_path
    settings["filename"] = model_name
    settings["path"] = config_path
    settings["filename_snn"] = "snn_"+model_name + \
                               "_"+str(int(settings["percentile"]))

    if normalize == "false":
        settings["normalize"] = False
        settings["evaluateANN"] = False

    snntoolbox.update_setup(settings)

    snntoolbox.test_full()

    print ("[MESSAGE] The experiment result is saved at %s" % (log_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running Max-Pooling \
                                     Experiments by Yuhuang Hu")
    parser.add_argument("-e", "--exp-name", type=str,
                        help="Experiment name.")
    parser.add_argument("-m", "--model-name", type=str,
                        help="The name of the model")
    parser.add_argument("-p", "--pref-name", type=str,
                        help="Destination of the json perf file.")
    parser.add_argument("-d", "--dataset", type=str,
                        help="type of the datset, mnist or cifar10")
    parser.add_argument("-n", "--normalize", type=str,
                        default="true",
                        help="no normalize if the model is normalized before")
    # as there is no setting parameters for this, not simply omit.
    parser.add_argument("--pool-type", type=str,
                        default="avg_max",
                        help="The type of max-pooling")
    args = parser.parse_args()
    maxpool_exp(**vars(args))
