from __future__ import print_function
import os
import keras
from keras import models, metrics
from keras.models import Sequential
from keras import optimizers
from keras.preprocessing import image
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.models import Model
import numpy as np
import time
from keras import backend as K

from snntoolbox.bin.run import main
from snntoolbox.utils.utils import import_configparser

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

n = 3
depth = n * 6

def l1_reg(weight_matrix):
    return 0.01 * K.mean(K.abs(weight_matrix))

def lr_schedule(epoch):
    lr = 1e-4
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 60:
        lr *= 1e-2
    elif epoch > 40:
        lr *= 1e-3
    print('Learning rate: ', lr)
    return lr

model_type = 'ResNet%dv' % (depth)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

path_wd = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(
    __file__)), '..', 'temp_imagenet', str(time.time())))
os.makedirs(path_wd)
dataset_wd = '/home/qinche/imagenet_validation'

model_name = 'imagenet_resnet'



train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

img_rows = 224
img_cols = 224
num_classes = 5

input_shape = (img_rows, img_cols, 3)

train_generator = train_datagen.flow_from_directory(
    # '/mnt/data/datasets/train',
    '/home/qinche/imagenet_train',
    target_size=(img_rows, img_cols),
    # The target_size is the size of your input images,every image will be resized to this size
    batch_size=64,
    class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(
    # '/mnt/data/datasets/validation',
    '/home/qinche/imagenet_validation',
    target_size=(img_rows, img_cols),
    # The target_size is the size of your input images,every image will be resized to this size
    batch_size=64,
    class_mode='categorical')


def resnet_layer(inputs,
                 num_filters=64,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):

    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(0.001))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes=num_classes):
    # if (depth - 2) % 6 != 0:
    #     raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    num_filters = 64
    num_res_blocks = 2

    # inputs = Input(shape=input_shape)
    # x = resnet_layer(inputs=inputs)
    inputs = Input(shape=input_shape)
    conv = Conv2D(64,
                  kernel_size=7,
                  strides=2,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(0.001))

    # x = conv(inputs)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    x = conv(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=(3,3), strides=2, padding='same')(x)
    x = resnet_layer(inputs=x)

    for stack in range(4):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides,
                             activation='relu')
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation='relu')
            if stack > 0 and res_block == 0:  # first layer but not first stack
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation='relu',
                                 batch_normalization=True)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2


    x = AveragePooling2D(pool_size=(1,1))(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    model = Model(inputs=inputs, outputs=outputs)
    return model


model = resnet_v1(input_shape=input_shape, depth=depth)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])
model.summary()
print(model_type)

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=2,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler]

datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-06,
        rotation_range=0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.,
        zoom_range=0.,
        channel_shift_range=0.,
        fill_mode='nearest',
        cval=0.,
        horizontal_flip=True,
        vertical_flip=False,
        rescale=None,
        preprocessing_function=None,
        data_format=None,
        validation_split=0.0)

# datagen.fit(train_generator)
# model2.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
model.fit_generator(
    train_generator,
    steps_per_epoch=None,
    shuffle=True,
    epochs=5,
    verbose=1,
    validation_data=validation_generator,
    workers=4,
    callbacks=callbacks
)

model_name = 'imagenet_resnet'
keras.models.save_model(model, os.path.join(path_wd, model_name + '.h5'))

score = model.evaluate_generator(validation_generator,
                                 verbose=1)
print("Evaluation Result of Original Model on ImageNet2012: " + str(score))

# SNN TOOLBOX CONFIGURATION #
#############################

# # Create a config file with experimental setup for SNN Toolbox.
# configparser = import_configparser()
# config = configparser.ConfigParser()
#
# config['paths'] = {
#     'path_wd': path_wd,             # Path to model.
#     'dataset_path': dataset_wd,        # Path to dataset.
#     'filename_ann': model_name      # Name of input model.
# }
#
# config['tools'] = {
#     'evaluate_ann': True,           # Test ANN on dataset before conversion.
#     'normalize': True               # Normalize weights for full dynamic range.
# }
#
# config['input]'] = {
#     'dataset_format': 'jpg',
#     'datagen_kwargs': {'preprocessing_function': 'helper_functions'},
#     'dataflow_kwargs': {'target_size': (224, 224), 'shuffle': True},
# }
# config['normalization'] = {
#     'network_mode': 1                 # 1 denotes the ResNet
# }
#
# config['simulation'] = {
#     'simulator': 'INI',             # Chooses execution backend of SNN toolbox.
#     'duration': 500,                 # Number of time steps to run each sample.
#     'num_to_test': 250,             # How many test samples to run.
#     'batch_size': 50,               # Batch size for simulation.
#     'keras_backend': 'tensorflow'   # Which keras backend to use.
# }
#
# config['output'] = {
#     'plot_vars': {                  # Various plots (slows down simulation).
#         # 'normalization_activations'
#         # 'spiketrains',              # Leave section empty to turn off plots.
#         # 'spikerates',
#         # 'activations',
#         # 'correlation',
#         # 'v_mem',
#         # 'error_t'
#         }
# }
#
# # Store config file.
# config_filepath = os.path.join(path_wd, 'config')
# with open(config_filepath, 'w') as configfile:
#     config.write(configfile)
#
# # RUN SNN TOOLBOX #
# ###################
#
# main(config_filepath)
