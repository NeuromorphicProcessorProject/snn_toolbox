"""Running experiments for max-pooling experiments.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

from __future__ import print_function
import os
import argparse
import json

import snntoolbox

try:
    from pushbullet import Pushbullet
    NOTIFICATION = True
    print ("[MESSAGE] Notification is turned on.")
except ImportError:
    NOTIFICATION = False
    print ("[MESSAGE] Notification is turned off.")

home_path = os.environ["HOME"]
config_path = os.path.join(home_path, ".snntoolbox")
pref_dir = os.path.join(config_path, "preferences")
log_dir = os.path.join(home_path, "workspace", "snntoolbox-log",
                       "pool-exps-new")
data_dir = os.path.join(config_path, "datasets")

if NOTIFICATION is True:
    notify_api_path = os.path.join(pref_dir, "api.txt")
    try:
        with open(notify_api_path, mode="r") as f:
            api_key = f.read().replace("\n", "")
        pb = Pushbullet(api_key)
    except IOError:
        NOTIFICATION = False
        print ("[MESSAGE] No valid API file is found.")


def maxpool_exp(exp_name, model_name, pref_name, dataset,
                normalize, online_normalize, pool_type,
                percentile):
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
    online_normalization : string
        true : use online normalization
        false : otherwise
    pool_type : str
        the name of the max pooling type
        "avg_max" or "fir_max"
    percentile : float
    """
    pref_path = os.path.join(pref_dir, pref_name)
    log_path = os.path.join(log_dir, exp_name)
    data_path = os.path.join(data_dir, dataset)

    if not os.path.exists(pref_path):
        raise ValueError("[MESSAGE] The target preference "
                         "file %s is not existed!" % pref_path)

    if not os.path.isdir(log_path):
        os.makedirs(log_path)

    print ("[MESSAGE] Running experiment %s." % exp_name)
    print ("[MESSAGE] Loading Experiment settings.")
    settings = json.load(open(pref_path))

    settings["log_dir_of_current_run"] = log_path
    settings["runlabel"] = exp_name
    settings["dataset_path"] = data_path
    settings["filename_ann"] = model_name
    settings["path_wd"] = config_path
    if percentile != 0.0:
        settings["percentile"] = percentile
    settings["filename_snn"] = "snn_"+model_name + \
                               "_"+str(settings["percentile"])

    if normalize == "false":
        settings["normalize"] = False
        settings["evaluate_ann"] = False

    if online_normalize == "false":
        settings["online_normalization"] = False

    settings["maxpool_type"] = pool_type

    # shutdown payloads
    settings["payloads"] = False

    snntoolbox.update_setup(settings)

    snntoolbox.test_full()

    end_message = "[MESSAGE] The experiment result is saved at %s" % log_path

    print (end_message)
    if NOTIFICATION is True:
        pb.push_note("Experiment %s Finished" % exp_name, end_message)

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
    parser.add_argument("-on", "--online-normalize", type=str,
                        default="true",
                        help="no normalize if the model is normalized before")
    # as there is no setting parameters for this, not simply omit.
    parser.add_argument("-pt", "--pool-type", type=str,
                        default="avg_max",
                        help="The type of max-pooling")
    parser.add_argument("-pc", "--percentile", type=float,
                        default=0.0,
                        help="The value of percentile.")
    args = parser.parse_args()
    maxpool_exp(**vars(args))
