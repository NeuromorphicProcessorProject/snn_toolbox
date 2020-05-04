import os
import keras
import tensorflow as tf
from keras.applications.mobilenet import preprocess_input, decode_predictions
from keras import optimizers
from keras.preprocessing import image
from keras.utils import to_categorical

from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import time

path_wd = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(
    __file__)), '..', 'temp_imagenet', str(time.time())))
os.makedirs(path_wd)

model = ResNet50(weights='imagenet')

model_name = 'imagenet_resnet'

opt = optimizers.Adam(lr=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
keras.models.save_model(model, os.path.join(path_wd, model_name + '.h5'))

validation_datagen = ImageDataGenerator()

img_rows = 224
img_cols = 224

input_shape = (img_rows, img_cols, 3)

validation_generator = validation_datagen.flow_from_directory(
    '/mnt/data/datasets/validation',
    # '/home/qinche/imagenet_validation',
    target_size=(img_rows, img_cols),
    # The target_size is the size of your input images,every image will be resized to this size
    batch_size=64,
    class_mode='categorical')


score = model.evaluate_generator(validation_generator,
                                 verbose=1)
print("Evaluation Result of Original Model on ImageNet2012: " + str(score))



# Train on ImageNet
# checkpoint_path = "ResNet/cp-{epoch:04d}.ckpt"
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
# print("Starting to train ResNet...")
#
# model.fit_generator(imagenet_generator(train_dataset, batch_size=Nbatch, is_training=True),
#                     steps_per_epoch=Ntrain // Nbatch,
#                     epochs=epochs,
#                     validation_data=imagenet_generator(validation_dataset, batch_size=Nbatch),
#                     validation_steps=Nvalidation // Nbatch,
#                     verbose=1,
#                     callbacks=[cp_callback, csv_logger])
