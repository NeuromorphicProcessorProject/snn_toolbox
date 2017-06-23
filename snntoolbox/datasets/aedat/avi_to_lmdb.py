# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 10:40:36 2016

@authors: Danny Neil, iulialexandra
@contact: iulialexandralungu@gmail.com

Utility file used to transform a series of AVI files into an LMDB for
training in Caffe.

Uses categories {paper, rock, scissors, background} for classification.
"""
import os
import os.path
import imageio
import caffe
import lmdb
import os
import argparse
import time
import sys
import numpy as np

DB_KEY_FORMAT = "{:0>10d}"


def create_label_files(label_filenames, num_frames, working_dir, labels_dir):
    """Creates one .txt file for each AVI movie containing the class number
    for each frame in the movie.

    Parameters
    ----------

    label_filenames: list of filenames, one for each AVI movie,
                     where the labels will be saved
    num_frames: list of frame number in each AVI movie.
    working_dir: working directory, where the labels files will be saved

    """
    labels_dir = os.path.join(working_dir, labels_dir)
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)
    for idx, f in enumerate(label_filenames):
        label = f.split("_")[0]
        fout = open(os.path.join(labels_dir, f), "w+")
        if (label.find("paper") == 0):
            for frame in range(num_frames[idx]):
                fout.write(" 0\n")
        if (label.find("scissors") == 0):
            for frame in range(num_frames[idx]):
                fout.write(" 1\n")
        if (label.find("rock") == 0):
            for frame in range(num_frames[idx]):
                fout.write(" 2\n")
        if (label.find("background") == 0):
            for frame in range(num_frames[idx]):
                fout.write(" 3\n")
        fout.close()


def create_workfile(dir_to_walk, working_dir, workfile, labels_dir):
    """Traverses a directory and its subdirectories and places all .avi files
    it finds in a .txt file, along with the name of a label file corresponding
    to each .avi.

    Parameters
    ----------

    dir_to_walk: directory to traverse in search for .avi files
    working_dir: directory where the workfile will be saved
    workfile: name of the workfile
    labels_dir: directory where the labels files will be saved
    """
    recordings = []
    num_frames = []
    label_filenames = []
    for dirpath, dirnames, filenames in os.walk(dir_to_walk):
        for filename in (f for f in filenames if f.endswith(".avi")):
            avi_path = os.path.join(dirpath, filename)
            recordings.append(avi_path)
            label_filenames.append(filename.split('.')[0] + '_label.txt')
            vid = imageio.get_reader(avi_path, 'ffmpeg')
            num_frames.append(vid._meta['nframes'])
            import math
            if math.isinf(vid._meta['nframes']):
                print("The following avi movie has too many frames: ",
                      label_filenames[-1])
                sys.exit()
    create_label_files(label_filenames, num_frames, working_dir, labels_dir)
    fout = open(os.path.join(working_dir, workfile), "w+")
    for idx, rec in enumerate(recordings):
        fout.write(rec + '    ' + os.path.join(working_dir, labels_dir,
                                               label_filenames[idx] + "\n"))
    fout.close()


def avi_to_frame_list(avi_filename, gray):
    """Creates a list of frames starting from an AVI movie.
    Inverts axes to have num_channels, height, width in this order.

    Parameters
    ----------

    avi_filename: name of the AVI movie
    gray: if True, the resulting images are treated as grey images with only
          one channel. If False, the images have three channels.
    """
    print('Loading {}'.format(avi_filename))
    vid = imageio.get_reader(avi_filename, 'ffmpeg')
    if gray:
        data = [np.mean(np.moveaxis(im, 2, 0), axis=0, keepdims=True)
                for im in vid.iter_data()]
        print('Loaded grayscale images.')

    else:
        data = [np.moveaxis(im, 2, 0) for im in vid.iter_data()]
        print('Loaded RGB images.')
    return data


def label_file_to_labels(label_filename):
    """Reads a file containing labels for one AVI movie and puts it in a list.
    """
    with open(label_filename, 'r') as f:
        all_labels = [int(label) for label in f.readlines()]
    return all_labels


def read_data_from_LMDB(read_db, max_to_read=None):
    """Reads data from LMDB database and returns a list of entries in
    non-human readable form. To convert them to string, use Caffe's tools
    """
    data = []
    with read_db.begin() as txn:
        if max_to_read is None:
            max_to_read = txn.stat()['entries']
        cursor = txn.cursor()
        it = cursor.iternext(keys=False, values=True)
        for counter in range(max_to_read):
            if counter % 100000 == 0:
                print("Reading entry {}".format(counter))
            data.append(it.item())
            it.next()
    read_db.close()
    return data


def shuffle_LMDB(in_lmdb_name, LMDB_path):
    """Creates shuffled LMDB starting from a given LMDB.
    """
    print("Shuffling database {}".format(in_lmdb_name))
    in_lmdb = lmdb.open(os.path.join(LMDB_path, in_lmdb_name), readonly=True)
    shuffled_db = lmdb.open(os.path.join(LMDB_path,
                                         'shuffled_' + in_lmdb_name))
    with in_lmdb.begin() as in_txn:
        num_entries = in_txn.stat()['entries']
        random_indices = np.arange(num_entries)
        np.random.shuffle(random_indices)

        with shuffled_db.begin(write=True) as shuffled_txn:
            for i in range(num_entries):
                if i % 100000 == 0:
                    print(i)
                in_key = DB_KEY_FORMAT.format(random_indices[i])
                in_dat = in_txn.get(in_key)
                out_key = DB_KEY_FORMAT.format(i)
                shuffled_txn.put(out_key, in_dat)

    in_lmdb.close()
    shuffled_db.close()


def write_data_to_lmdb(db, data_in, labels_in, curr_idx):
    """Given arrays of data and the labels, it writes the information in an
    LMDB

    Parameters
    ----------

    db: LMDB to write the data into
    data_in: image arrays
    labels_in: list of labels
    curr_idx: the idx used as key for writing in the LMDB
    """
    with db.begin(write=True) as in_txn:
        for i in range(len(data_in)):
            d, l = data_in[i], labels_in[i]
            im_dat = caffe.io.array_to_datum(d.astype('uint8'),
                                             label=int(l))
            key = DB_KEY_FORMAT.format(curr_idx)
            in_txn.put(key, im_dat.SerializeToString())
            curr_idx += 1
    return curr_idx


def write_categ_lmdb(workfile, categories, LMDB_path, gray):
    """Given a workfile containing the AVI movies and the labels for each
    frame, this function creates an LMDB for each classification category.
    """
    # Open databases
    categ_db = []
    for idx, categ in enumerate(categories):
        categ_db.append(
            lmdb.open(os.path.join(LMDB_path, categ), map_size=int(1e12),
                      map_async=True, writemap=True, meminit=False))
    curr_idx = np.zeros(len(categories), dtype='int')
    with open(workfile, 'r') as f:
        for line in f.readlines():
            # Load work to do
            avi_file, label_file = line.strip().split('    ')

            # Convert to data
            file_frames = avi_to_frame_list(avi_file, gray)
            file_labels = label_file_to_labels(label_file)

            # Quick check the lengths
            assert len(file_frames) == len(file_labels), \
                'Frames and Labels do not match in length!'

            # Write data in each LMDB corresponding to the .avi category
            db_label = avi_file.split("/")[-1].split("_")[0]
            index = categories.index(db_label)
            curr_idx[index] = write_data_to_lmdb(categ_db[index],
                                                 file_frames, file_labels,
                                                 curr_idx[index])
    return curr_idx


def write_train_test_lmdb(categories, LMDB_path, train_test_split, num_rot):
    """Entire pipeline of train and test LMDB creation.
    """
    DB_KEY_FORMAT = "{:0>10d}"
    curr_idx = np.zeros(len(categories), dtype='int')
    for c_idx, categ in enumerate(categories):
        categ_db = lmdb.open(os.path.join(LMDB_path, categ), readonly=True)
        with categ_db.begin() as txn:
            curr_idx[c_idx] = txn.stat()['entries']
        categ_db.close()
    min_idx = min(curr_idx)
    print("Number of images for each category: ", curr_idx)

    train_idx = int(train_test_split * min_idx)
    print("train samples ", train_idx)
    test_idx = min_idx - train_idx
    print("test samples ", test_idx)

    indices_categ = np.arange(min_idx)
    print(len(categories))
    indices_train_database = np.arange(
        len(categories) * train_idx * (num_rot + 1))
    print("total training samples ", len(indices_train_database))
    np.random.shuffle(indices_train_database)
    indices_test_database = np.arange(
        len(categories) * test_idx * (num_rot + 1))
    print("total test samples", len(indices_test_database))
    np.random.shuffle(indices_test_database)

    # Open databases
    train_db = lmdb.open(os.path.join(LMDB_path, 'train'), map_size=int(1e12),
                         map_async=True, writemap=True, meminit=False)
    test_db = lmdb.open(os.path.join(LMDB_path, 'test'), map_size=int(1e12),
                        map_async=True, writemap=True, meminit=False)
    last = time.time()
    # Iterate over the category LMDBs
    curr_train_idx, curr_test_idx = 0, 0
    for idx, categ in enumerate(categories):
        categ_db = lmdb.open(os.path.join(LMDB_path, categ), readonly=True)
        with categ_db.begin() as categ_txn:
            print("Writing {} samples to database".format(categ))
            np.random.shuffle(indices_categ)
            # Populating training LMDB
            with train_db.begin(write=True) as in_txn_train:
                for i in range(train_idx):
                    if i % 100000 == 0:
                        elapsed = time.time() - last
                        last = time.time()
                        print(
                            "Wrote {} training samples to database in {} "
                            "seconds".format(
                                i * (num_rot + 1),
                                elapsed))
                    in_idx = indices_categ[i]
                    in_key = DB_KEY_FORMAT.format(in_idx)
                    in_item = categ_txn.get(in_key)

                    out_key = DB_KEY_FORMAT.format(indices_train_database[i * (
                    num_rot + 1) + curr_train_idx])
                    in_txn_train.put(out_key, in_item)

                    # Rotate images by 90 degrees
                    datum = caffe.proto.caffe_pb2.Datum()
                    datum.ParseFromString(in_item)
                    assert datum.channels == 1, "The algorithm currently only " \
                                                "works for 1-channel images"
                    flat_x = np.fromstring(datum.data, dtype=np.uint8)
                    x = flat_x.reshape(datum.height, datum.width)
                    y = datum.label
                    for rotation_idx in range(num_rot):
                        x = np.rot90(x)
                        out_image = np.reshape(x, (
                        datum.channels, datum.height, datum.width))
                        im_dat = caffe.io.array_to_datum(
                            out_image.astype('uint8'),
                            label=int(y))
                        out_key = DB_KEY_FORMAT.format(indices_train_database[
                                                           i * (
                                                           num_rot + 1) +
                                                           curr_train_idx
                                                           + rotation_idx + 1])
                        in_txn_train.put(out_key, im_dat.SerializeToString())
                curr_train_idx += train_idx * (num_rot + 1)

            with test_db.begin(write=True) as in_txn_test:
                for i in range(test_idx):
                    if i % 100000 == 0:
                        print("Wrote {} testing samples to database".format(
                            i * (num_rot + 1)))
                    in_idx = indices_categ[train_idx + i]
                    in_key = DB_KEY_FORMAT.format(in_idx)
                    in_item = categ_txn.get(in_key)

                    out_key = DB_KEY_FORMAT.format(indices_test_database[i * (
                    num_rot + 1) + curr_test_idx])
                    in_txn_test.put(out_key, in_item)

                    datum = caffe.proto.caffe_pb2.Datum()
                    datum.ParseFromString(in_item)
                    assert datum.channels == 1, "The algorithm currently only " \
                                                "works for 1-channel images"
                    flat_x = np.fromstring(datum.data, dtype=np.uint8)
                    x = flat_x.reshape(datum.height, datum.width)
                    y = datum.label

                    for rotation_idx in range(num_rot):
                        x = np.rot90(x)
                        out_image = np.reshape(x, (
                        datum.channels, datum.height, datum.width))
                        im_dat = caffe.io.array_to_datum(
                            out_image.astype('uint8'),
                            label=int(y))
                        out_key = DB_KEY_FORMAT.format(indices_test_database[
                                                           i * (
                                                           num_rot + 1) +
                                                           curr_test_idx +
                                                           rotation_idx + 1])
                        in_txn_test.put(out_key, im_dat.SerializeToString())

                curr_test_idx += test_idx * (num_rot + 1)

            print('Wrote so far: {} train, {} test.\n'.format(curr_train_idx,
                                                              curr_test_idx))

        categ_db.close()
    test_db.close()
    train_db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Turn a list of movies into\
                                                  an LMDB.')
    parser.add_argument('--seed', default=42, type=int, help='Initialize the\
                        random seed of the run (for reproducibility).')
    parser.add_argument('--workfile', default='workfile_train.txt',
                        help='File which is a list of movie files\
                        and labels to process,\
                        one per line, separated by four spaces')
    parser.add_argument('--categories', default=['paper', 'scissors', 'rock',
                                                 'background'],
                        help='List of categories used for classification')
    parser.add_argument('--LMDB_path', default='./lmdb_train/',
                        help='Where to write out the LMDB dataset.')
    parser.add_argument('--avi_dir', default='./recordings',
                        help='Where to look for .avi files to add to the LMDB')
    parser.add_argument('--working_dir', default='.',
                        help='Where to create the workfile used for creating '
                             'the LMDB')
    parser.add_argument('--labels_dir', default='labels_train',
                        help='Where to create the labels files')
    parser.add_argument('--train_test_split', default=0.8, type=float,
                        help='Where to split the data (not shuffled).')
    parser.add_argument('--num_rotations', default=3, type=int,
                        help='How many 90 degree rotations to perform for '
                             'each image')
    parser.add_argument('--gray', default=True,
                        help='If the input data is grayscale, the output' +
                             ' LMDB images will have only one channel. '
                             'Otherwise' +
                             ' it will have 3 channels.')
    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)

    # Make the output directory
    if not os.path.exists(args.LMDB_path):
        os.makedirs(args.LMDB_path)

    # Make the workfile used to create the LMDBs
    create_workfile(args.avi_dir, args.working_dir, args.workfile,
                    args.labels_dir)

    # Create the LMDBs
    start_time = time.time()
    curr_idx = write_categ_lmdb(args.workfile, args.categories,
                                args.LMDB_path, args.gray)
    print("Number of samples for each category:{}".format(curr_idx))

    write_train_test_lmdb(args.categories, args.LMDB_path,
                          args.train_test_split, args.num_rotations)
    print('Finished in {}s.'.format(time.time() - start_time))