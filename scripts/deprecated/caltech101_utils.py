# -*- coding: utf-8 -*-
"""

Adapted from a bachelor project of Marco Unternaehrer at INI.

"""

from __future__ import absolute_import, print_function

import os
import sys
import scipy
import tarfile
import json
import re
from six.moves.urllib.request import FancyURLopener
import numpy as np

import snntoolbox


# Configuration
url_original = "http://www.vision.caltech.edu/Image_Datasets/Caltech101/" + \
               "101_ObjectCategories.tar.gz"
url_img_gen = "https://www.googledrive.com/host/0B6t56IB_eb6hNjY1NXFSZEdsbkE"
url_img_gen_resized = \
    "https://www.googledrive.com/host/0B6t56IB_eb6hdXliTUFMdzZEMU0"
tar_inner_dirname = "101_ObjectCategories"
data_dir = os.path.join(snntoolbox.toolbox_root, 'datasets', 'caltech101')
nb_classes = 102


def get_file(url, destination_dir, fname):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    file_dest = os.path.join(destination_dir, fname)

    try:
        open(file_dest)
    except:
        print('Downloading data from', url)
        FancyURLopener().retrieve(url, file_dest)

    return file_dest


def untar_file(fpath, untar_dir, dataset):
    if not os.path.exists(os.path.join(untar_dir, dataset)):
        print('Untaring file...')
        tfile = tarfile.open(fpath, 'r:gz')
        tfile.extractall(path=untar_dir)
        tfile.close()

    return os.path.join(untar_dir, dataset)


def make_relative_path(full_path):
    base_path, fname_ext = os.path.split(full_path)
    parent_dir = os.path.split(base_path)[1]
    return os.path.join(parent_dir, fname_ext)


def load_samples(fpaths, nb_samples=None):
    if nb_samples is None:
        nb_samples = len(fpaths)

    # determine height / width
    img = load_img(fpaths[0])
    (width, height) = img.size

    # allocate memory
    sample_data = np.zeros((nb_samples, 3, height, width), dtype="uint8")

    counter = 0
    for i in range(nb_samples):
        img = load_img(fpaths[i])
        r, g, b = img.split()
        sample_data[counter, 0, :, :] = np.array(r)
        sample_data[counter, 1, :, :] = np.array(g)
        sample_data[counter, 2, :, :] = np.array(b)
        counter += 1

    return sample_data


def load_paths_from_dir(base_path, full_path=True):
    X_dict = create_label_path_dict(base_path, full_path=full_path)
    X_paths, y = split_label_path_dict(X_dict)
    return X_paths, y


def load_paths_from_files(base_path, fname_x, fname_y, full_path=True):
    X_path = os.path.abspath(os.path.join(base_path, '..', fname_x))
    y_path = os.path.abspath(os.path.join(base_path, '..', fname_y))
    if os.path.isfile(X_path) and os.path.isfile(y_path):
        X = [s[2:-1] for s in np.loadtxt(X_path, dtype=np.str)]
        if full_path:
            X = np.array([os.path.join(base_path, p) for p in X])
        y = np.loadtxt(y_path, dtype=np.int)

        return X, y
    else:
        raise Exception


def create_label_path_dict(base_path, full_path=False, seed=None):
    label_path_dict = {}

    # directories are the labels
    labels = sorted([d for d in os.listdir(base_path)])

    # loop over all subdirs
    for label_class_nr, label in enumerate(labels):
        label_dir = os.path.join(base_path, label)
        fpaths = np.array([img_fname for img_fname in
                           list_pictures(label_dir)])
        if not full_path:
            fpaths = np.array(map(make_relative_path, fpaths))

        if seed:
            np.random.seed(seed)
        np.random.shuffle(fpaths)

        stacked = np.dstack((fpaths,
                             [label_class_nr for x in range(len(fpaths))]))[0]
        label_path_dict[label_class_nr] = stacked

    return label_path_dict


def split_label_path_dict(label_path_dict):
    path_label = np.concatenate(label_path_dict.values(), axis=0)
    swap = np.swapaxes(path_label, 0, 1)
    paths = swap[0]
    labels = swap[1]

    return paths, labels


def convert_to_label_path_dict(path_label_array):
    label_path_dict = {}

    for path, label in path_label_array:
        if label not in label_path_dict:
            label_path_dict[label] = []

        label_path_dict[label] += [np.array([path, label])]

    return label_path_dict


def train_test_split(label_path_dict, y=None, test_size=0.2, stratify=True,
                     seed=None):

    dict_input = True
    if type(label_path_dict) != dict:
        dict_input = False
        label_path_dict = convert_to_label_path_dict(zip(label_path_dict, y))

    if stratify:
        train_dict = {}
        test_dict = {}

        for label, path_label_array in label_path_dict.iteritems():

            if seed:
                np.random.seed(seed)
            np.random.shuffle(path_label_array)

            if test_size < 1:
                # test_size is split ratio
                nb_train_items = int(len(path_label_array) * (1.0 - test_size))
            else:
                # test_size is number of images per category
                nb_train_items = len(path_label_array) - test_size

            train_dict[label] = path_label_array[:nb_train_items]
            test_dict[label] = path_label_array[nb_train_items:]
    else:
        path_label_array = np.concatenate(label_path_dict.values(), axis=0)

        if seed:
            np.random.seed(seed)
        np.random.shuffle(path_label_array)

        if test_size < 1:
            # test_size is split ratio
            nb_train_items = int(len(path_label_array) * (1.0 - test_size))
        else:
            # test_size is number of images per category
            nb_train_items = len(path_label_array) - test_size

        train_dict = convert_to_label_path_dict(path_label_array[
                                                            :nb_train_items])
        test_dict = convert_to_label_path_dict(path_label_array[
                                                            nb_train_items:])

    if not dict_input:
        X_train, y_train = split_label_path_dict(train_dict)
        X_test, y_test = split_label_path_dict(test_dict)

        assert np.intersect1d(X_train, X_test).size == 0

        return (X_train, y_train), (X_test, y_test)
    else:
        assert np.intersect1d(split_label_path_dict(train_dict)[0],
                              split_label_path_dict(test_dict)[0]).size == 0

        return train_dict, test_dict


def already_split(base_path, test_size, stratify, seed):
    split_config_path = os.path.abspath(os.path.join(base_path, '..',
                                                     'split_config.txt'))

    if os.path.isfile(split_config_path):
        with open(split_config_path) as data_file:
            split_config = json.load(data_file)

            same_config = float(test_size) == float(split_config['test_size'])\
                and stratify == bool(split_config['stratify'])

            same_seed = (seed == int(split_config['seed'])) if \
                split_config['seed'] else (seed == split_config['seed'])

            return same_config and same_seed

    return False


def load_train_test_split_paths(base_path):
    return load_paths_from_files(base_path, 'X_train.txt', 'y_train.txt'), \
           load_paths_from_files(base_path, 'X_test.txt', 'y_test.txt')


def load_data(path="", dataset='original', resize=False, width=240, height=180,
              test_size=0, stratify=True, seed=None):
    if not path:
        untar_dir = download(dataset=dataset)
        path = os.path.join(untar_dir, tar_inner_dirname)

    if resize:
        output_dir = os.path.join(data_dir, '{}-resized'.format(dataset),
                                  tar_inner_dirname)
        path = resize_imgs(input_dir=path, output_dir=output_dir,
                           target_width=width, target_height=height)

    if test_size:
        (X_train, y_train), (X_test, y_test) = load_paths(path=path,
                                                          test_size=test_size,
                                                          stratify=stratify,
                                                          full_path=True,
                                                          seed=seed)
        X_train = load_samples(X_train, len(X_train))
        X_test = load_samples(X_test, len(X_test))

        return (X_train, y_train), (X_test, y_test)
    else:
        X_paths, y = load_paths_from_dir(path, full_path=True)
        X = load_samples(X_paths, len(X_paths))

        return X, y


def load_paths(path="", dataset='img-gen-resized', test_size=0, stratify=True,
               full_path=True, seed=None):
    if not path:
        untar_dir = download(dataset=dataset)
        path = os.path.join(untar_dir, tar_inner_dirname)

    if test_size:
        if already_split(path, test_size, stratify, seed):
            print("Load train/test split from disk...")
            return load_train_test_split_paths(path)
        else:
            X_dict = create_label_path_dict(path, full_path=full_path,
                                            seed=seed)

            # generate split
            print("Generate train/test split...")
            train_dict, test_dict = train_test_split(X_dict,
                                                     test_size=test_size,
                                                     stratify=stratify)

            X_train, y_train = split_label_path_dict(train_dict)
            X_test, y_test = split_label_path_dict(test_dict)

            return (X_train, y_train), (X_test, y_test)
    else:
        X_paths, y = load_paths_from_dir(path, full_path=full_path)
        return X_paths, y


def load_cv_split_paths(base_path, cv_fold, full_path=True):
        return load_paths_from_files(base_path,
                                     'cv{}_X_train.txt'.format(cv_fold),
                                     'cv{}_y_train.txt'.format(cv_fold),
                                     full_path=full_path), \
               load_paths_from_files(base_path,
                                     'cv{}_X_test.txt'.format(cv_fold),
                                     'cv{}_y_test.txt'.format(cv_fold),
                                     full_path=full_path)


def download(destination_dir=data_dir, dataset='original'):
    if dataset == 'original':
        fname = 'original.tar.gz'
        url = url_original
    elif dataset == 'img-gen':
        fname = 'img-gen.tar.gz'
        url = url_img_gen
    elif dataset == 'img-gen-resized':
        fname = 'img-gen-resized.tar.gz'
        url = url_img_gen_resized

    local_tar_file = get_file(url, destination_dir, fname)
    untar_dir = untar_file(local_tar_file, destination_dir, dataset)

    return untar_dir


def array_to_img(x, scale=True):
    from PIL import Image
    x = x.transpose(1, 2, 0)
    if scale:
        x += max(-np.min(x), 0)
        x /= np.max(x)
        x *= 255
    if x.shape[2] == 3:
        # RGB
        return Image.fromarray(x.astype("uint8"), "RGB")
    else:
        # grayscale
        return Image.fromarray(x[:, :, 0].astype("uint8"), "L")


def img_to_array(img):
    x = np.asarray(img, dtype='float32')
    if len(x.shape) == 3:
        # RGB: height, width, channel -> channel, height, width
        x = x.transpose(2, 0, 1)
    else:
        # grayscale: height, width -> channel, height, width
        x = x.reshape((1, x.shape[0], x.shape[1]))
    return x


def load_img(path, grayscale=False):
    from PIL import Image
    img = Image.open(path)
    if grayscale:
        img = img.convert('L')
    else:  # Assure 3 channel even when loaded image is grayscale
        img = img.convert('RGB')
    return img


def list_pictures(directory, ext='jpg|jpeg|bmp|png'):
    return [os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and
            re.match('([\w]+\.(?:' + ext + '))', f)]


def resize_imgs(input_dir, output_dir, target_width, target_height, quality=90,
                verbose=1):
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("starting....")
        print("Collecting data from %s " % input_dir)
        for subdir in os.listdir(input_dir):
            input_subdir = os.path.join(input_dir, subdir)
            output_subdir = os.path.join(output_dir, subdir)
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)
            if os.path.exists(output_subdir):
                for img_path in list_pictures(input_subdir):
                    try:
                        if verbose > 0:
                            print("Resizing file : %s " % img_path)

                        img = load_img(img_path)
                        zoom_factor = min(float(target_width) / img.width,
                                          float(target_height) / img.height)

                        img = img_to_array(img)

                        img = scipy.ndimage.interpolation.zoom(
                            img, zoom=(1., zoom_factor, zoom_factor))

                        (_, height, width) = img.shape
                        pad_h_after = (target_height - height) / 2
                        pad_h_before = int(np.ceil(pad_h_after))
                        pad_w_after = (target_width - width) / 2
                        pad_w_before = int(np.ceil(pad_w_after))
                        img = np.pad(img, ((0, 0), (pad_h_before, pad_h_after),
                                           (pad_w_before, pad_w_after)),
                                     mode='edge')

                        img = array_to_img(img)

                        _, fname_ext = os.path.split(img_path)
                        out_file = os.path.join(output_dir, subdir, fname_ext)
                        img.save(out_file, img.format, quality=quality)
                        img.close()
                    except Exception:
                        print("Error resize file : %s - %s " % (subdir,
                                                                img_path))

    except Exception as e:
        print("Error, check Input directory etc : ", e)
        sys.exit(1)
    return output_dir
