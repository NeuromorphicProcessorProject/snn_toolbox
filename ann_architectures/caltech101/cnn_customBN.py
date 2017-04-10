from __future__ import absolute_import
from __future__ import print_function
import json
import datetime
import os
import numpy as np
#np.random.seed(42) # make keras deterministic

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.callbacks import CallbackList, ModelCheckpoint
from keras.regularizers import l2

from sklearn.utils import compute_class_weight

from ini_caltech101 import util
from ini_caltech101.dataset import caltech101
from ini_caltech101.keras_extensions.constraints import zero
from ini_caltech101.keras_extensions.callbacks import INIBaseLogger, INILearningRateScheduler, INILearningRateReducer, INIHistory
from ini_caltech101.keras_extensions.schedules import TriangularLearningRate
from ini_caltech101.keras_extensions.normalization import BatchNormalization
from ini_caltech101.keras_extensions.optimizers import INISGD
'''
    Train a CNN on a data augmented version of the Caltech101 images dataset.
    GPU run command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python caltech101_cnn_training.py
'''

##########################
# GENERAL PARAMETERS
##########################
batch_size = 64
nb_epoch = 1
nb_classes = caltech101.config.nb_classes

shuffle_data = True
normalize_data = True
batch_normalization = True

b_constraint = zero() # bias constraint: zero() or None

# shape of the image (SHAPE x SHAPE)
image_width, image_height = 240, 180

# the caltech101 images are RGB
image_dimensions = 3

##########################
# DATA LOADING
##########################

print("Loading paths...")

# download & untar or get local path
base_path = caltech101.download(dataset='img-gen-resized')

# path to image folder
base_path = os.path.join(base_path, caltech101.config.tar_inner_dirname)

# X_test contain only paths to images
(X_test, y_test) = util.load_paths_from_files(base_path, 'X_test.txt', 'y_test.txt')

for cv_fold in [0]: # on which cross val folds to run; cant loop over several folds due to some bug
    print("fold {}".format(cv_fold))

    experiment_name = '_bn_triangular_cv{}_e{}'.format(cv_fold, nb_epoch)

    # load cross val split
    (X_train, y_train), (X_val, y_val) = util.load_cv_split_paths(base_path, cv_fold)

    # compute class weights, since classes are highly imbalanced
    class_weight = compute_class_weight('auto', range(nb_classes), y_train)

    if normalize_data:
        print("Load mean and std...")
        X_mean, X_std = util.load_cv_stats(base_path, cv_fold)
        normalize_data = (X_mean, X_std)

    nb_train_sample = X_train.shape[0]
    nb_val_sample = X_val.shape[0]
    nb_test_sample = X_test.shape[0]

    print('X_train shape:', X_train.shape)
    print(nb_train_sample, 'train samples')
    if X_val is not None:
        print(nb_val_sample, 'validation samples')
    print(nb_test_sample, 'test samples')

    # shuffle/permutation
    if shuffle_data:
        (X_train, y_train) = util.shuffle_data(X_train, y_train, seed=None)
        (X_val, y_val) = util.shuffle_data(X_val, y_val, seed=None)
        (X_test, y_test) = util.shuffle_data(X_test, y_test, seed=None)

    ##########################
    # MODEL BUILDING
    ##########################

    print("Building model...")

    if batch_normalization:
        weight_reg = 5e-4 # weight regularization value for l2
        dropout = False
        dropout_fc_layer = False
        lr = 0.005
        lr_decay = 5e-4

    else:
        weight_reg = 5e-4 # weight regularization value for l2
        dropout = True
        lr = 0.005
        lr_decay = 5e-4

    model = Sequential()
    conv1 = Conv2D(128, 5, 5,
                          strides=(2, 2),
                          b_constraint=b_constraint,
                          init='he_normal',
                          W_regularizer=l2(weight_reg),
                          input_shape=(image_dimensions, image_width, image_height))
    model.add(conv1)
    if batch_normalization:
        model.add(BatchNormalization(mode=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    if dropout:
        model.add(Dropout(0.35))

    conv2 = Conv2D(256, 3, 3, b_constraint=b_constraint, init='he_normal', W_regularizer=l2(weight_reg))
    model.add(conv2)
    if batch_normalization:
        model.add(BatchNormalization(mode=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    if dropout:
        model.add(Dropout(0.35))

    model.add(ZeroPadding2D(padding=(1, 1)))
    conv3 = Conv2D(512, 3, 3, b_constraint=b_constraint, init='he_normal', W_regularizer=l2(weight_reg))
    model.add(conv3)
    if batch_normalization:
        model.add(BatchNormalization(mode=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    if dropout:
        model.add(Dropout(0.35))

    model.add(Flatten())

    model.add(Dense(1024, b_constraint=b_constraint, init='he_normal', W_regularizer=l2(weight_reg)))
    if batch_normalization:
        model.add(BatchNormalization(mode=1))
    model.add(Activation('relu'))

    if dropout or dropout_fc_layer:
        model.add(Dropout(0.5))

    model.add(Dense(nb_classes, b_constraint=b_constraint, init='he_normal', W_regularizer=l2(weight_reg)))
    model.add(Activation('softmax'))

    print('Compiling model...')
    sgd = INISGD(lr=lr, decay=lr_decay, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    #model.load_weights('results/2015-12-12_18.15.01_no-bn_lr-0.001_e20_img-gen_weights.hdf5')

    ##########################
    # TRAINING PREPARATIONS
    ##########################
    callbacks = []
    history = INIHistory()
    callbacks += [history]

    logger = INIBaseLogger()
    callbacks += [logger]

    step_size = 8 * (nb_train_sample / batch_size) # according to the paper: 2 - 8 times the iterations per epoch
    schedule = TriangularLearningRate(lr=0.001, step_size=step_size, max_lr=0.02, max_to_min=True)
    lrs = INILearningRateScheduler(schedule, mode='batch', logger=logger)
    callbacks += [lrs]

    #mcp = ModelCheckpoint('results/experiment' + experiment_name + '_epoch{epoch}_weights.hdf5', save_best_only=True)
    #callbacks += [mcp]

    #lrr = INILearningRateReducer(monitor='val_acc', improve='increase', decrease_factor=0.1, patience=3, stop=3, verbose=1)
    #callbacks += [lrr]

    callbacks = CallbackList(callbacks)

    shuffle_on_epoch_start = True
    metrics = ['loss', 'acc', 'val_loss', 'val_acc', 'val_class_acc'] # show those at epoch end
    do_validation = True

    callbacks._set_model(model)
    callbacks._set_params({
        'batch_size': batch_size,
        'nb_epoch': nb_epoch,
        'nb_sample': nb_train_sample,
        'verbose': 1,
        'do_validation': do_validation,
        'metrics': metrics,
    })

    ##########################
    # TRAINING
    ##########################
    callbacks.on_train_begin()

    model.stop_training = False
    for epoch in range(nb_epoch):
        callbacks.on_epoch_begin(epoch)

        if shuffle_on_epoch_start:
            X_train, y_train = util.shuffle_data(X_train, y_train)

        # train
        util.train_on_batch(model, X_train, y_train, nb_classes,
                            callbacks=callbacks,
                            normalize=normalize_data,
                            batch_size=batch_size,
                            class_weight=class_weight,
                            shuffle=False)

        epoch_logs = {}

        ##########################
        # VALIDATION
        ##########################
        if do_validation:
            # calculates the overall loss and accuracy
            val_loss, val_acc, val_size = util.test_on_batch(model, X_val, y_val, nb_classes,
                                                             normalize=normalize_data,
                                                             batch_size=batch_size,
                                                             shuffle=False)
            epoch_logs['val_loss'] = val_loss
            epoch_logs['val_acc'] = val_acc
            epoch_logs['val_size'] = val_size

            # calculates the accuracy per class
            class_acc = util.calc_class_acc(model, X_val, y_val, nb_classes,
                                            normalize=normalize_data,
                                            batch_size=batch_size,
                                            keys=['acc'])
            epoch_logs['val_class_acc'] = class_acc['acc']

        callbacks.on_epoch_end(epoch, epoch_logs)
        if model.stop_training:
            break

    training_end_logs = {}

    ##########################
    # TESTING
    ##########################
    test_loss, test_acc, test_size = util.test_on_batch(model, X_test, y_test, nb_classes,
                                                        normalize=normalize_data,
                                                        batch_size=batch_size,
                                                        shuffle=False)

    training_end_logs['test_loss'] = test_loss
    training_end_logs['test_acc'] = test_acc
    training_end_logs['test_size'] = test_size

    class_acc = util.calc_class_acc(model, X_test, y_test, nb_classes,
                                    normalize=normalize_data,
                                    batch_size=batch_size,
                                    keys=['acc'])
    training_end_logs['test_class_acc'] = class_acc['acc']

    callbacks.on_train_end(logs=training_end_logs)

    ##########################
    # SAVING
    ##########################
    dt = datetime.datetime.now()
    with open('{:%Y-%m-%d_%H.%M.%S}{}_architecture.json'.format(dt, experiment_name), 'w') as f:
        f.write(model.to_json())
    with open('{:%Y-%m-%d_%H.%M.%S}{}_history.json'.format(dt, experiment_name), 'w') as f:
        f.write(json.dumps(history.history))
    model.save_weights('{:%Y-%m-%d_%H.%M.%S}{}_weights.hdf5'.format(dt, experiment_name))