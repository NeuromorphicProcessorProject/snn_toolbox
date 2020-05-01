import os
import math
import numpy as np
# import cv2 as cv
import keras
import tensorflow as tf

from keras.applications import mobilenet
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input, decode_predictions
from keras import optimizers
from keras.preprocessing import image
from keras.utils import to_categorical

# import tensorflow_datasets as tfds

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import time

path_wd = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(
    __file__)), '..', 'temp_imagenet', str(time.time())))
os.makedirs(path_wd)

model = ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

model_name = 'imagenet_resnet'

# Load MobileNet model
#model = MobileNet(weights='imagenet')
opt = optimizers.Adam(lr=0.001)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
keras.models.save_model(model, os.path.join(path_wd, model_name + '.h5'))
# # Fetch the dataset directly
# imagenet = tfds.image.Imagenet2012()
# ## or by string name
# # imagenet = tfds.builder('imagenet2012')
#
# # Describe the dataset with DatasetInfo
# C = imagenet.info.features['label'].num_classes
# Ntrain = imagenet.info.splits['train'].num_examples
# Nvalidation = imagenet.info.splits['validation'].num_examples
# Nbatch = 32
# assert C == 1000
# assert Ntrain == 1281167
# assert Nvalidation == 50000
#
# # Download the data, prepare it, and write it to disk
# imagenet.download_and_prepare()
#
# # Load data from disk as tf.data.Datasets
# datasets = imagenet.as_dataset()
# train_dataset, validation_dataset = datasets['train'], datasets['validation']
# assert isinstance(train_dataset, tf.data.Dataset)
# assert isinstance(validation_dataset, tf.data.Dataset)
#
#
# def imagenet_generator(dataset, batch_size=32, num_classes=1000, is_training=False):
#     images = np.zeros((batch_size, 224, 224, 3))
#     labels = np.zeros((batch_size, num_classes))
#     while True:
#         count = 0
#         for sample in tfds.as_numpy(dataset):
#             image = sample["image"]
#             label = sample["label"]
#
#             images[count % batch_size] = mobilenet.preprocess_input(np.expand_dims(cv.resize(image, (224, 224)), 0))
#             labels[count % batch_size] = np.expand_dims(to_categorical(label, num_classes=num_classes), 0)
#
#             count += 1
#             if (count % batch_size == 0):
#                 yield images, labels
#
#
# # Infer on ImageNet
# labels = np.zeros((Nvalidation))
# pred_labels = np.zeros((Nvalidation, C))
# pred_labels_new = np.zeros((Nvalidation, C))
#
# score = model.evaluate_generator(imagenet_generator(validation_dataset, batch_size=32),
#                                  steps=Nvalidation // Nbatch,
#                                  verbose=1)
# print("Evaluation Result of Original Model on ImageNet2012: " + str(score))
#
# # Train on ImageNet
# checkpoint_path = "Mobilenet/cp-{epoch:04d}.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)
# os.makedirs(checkpoint_dir, exist_ok=True)
#
# cp_callback = keras.callbacks.ModelCheckpoint(
#     checkpoint_path, verbose=1, save_weights_only=True,
#     # Save weights, every 1-epoch
#     period=1)
#
# csv_logger = keras.callbacks.CSVLogger('MobileNet_training.csv')
#
# print("Starting to train Modified MobileNet...")
# epochs = 5
#
# model.fit_generator(imagenet_generator(train_dataset, batch_size=Nbatch, is_training=True),
#                     steps_per_epoch=Ntrain // Nbatch,
#                     epochs=epochs,
#                     validation_data=imagenet_generator(validation_dataset, batch_size=Nbatch),
#                     validation_steps=Nvalidation // Nbatch,
#                     verbose=1,
#                     callbacks=[cp_callback, csv_logger])
#
# model.save("MobileNet.h5")