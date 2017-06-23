# -*- coding: utf-8 -*-

"""
VGG16 model for Keras, from https://github.com/fchollet/deep-learning-models.

Reference:

[Very Deep Convolutional Networks for Large-Scale Image Recognition]
(https://arxiv.org/abs/1409.1556)

"""

from __future__ import print_function

from keras.preprocessing.image import ImageDataGenerator

from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Flatten, Dense, Conv2D, AveragePooling2D
from keras.models import Sequential
from keras.optimizers import SGD


def get_vgg16():
    """Build VGG16.

    Returns
    -------

    """

    model = Sequential()

    # Block 1
    model.add(Conv2D(64, 3, 3, padding='same', name='block1_conv1',
                            input_shape=(3, 224, 224), activation='relu'))
    model.add(Conv2D(64, 3, 3, padding='same', name='block1_conv2',
                            activation='relu'))
    model.add(AveragePooling2D(name='block1_pool'))

    # Block 2
    model.add(Conv2D(128, 3, 3, padding='same',
                            name='block2_conv1', activation='relu'))
    model.add(Conv2D(128, 3, 3, padding='same',
                            name='block2_conv2', activation='relu'))
    model.add(AveragePooling2D(name='block2_pool'))

    # Block 3
    model.add(Conv2D(256, 3, 3, padding='same',
                            name='block3_conv1', activation='relu'))
    model.add(Conv2D(256, 3, 3, padding='same',
                            name='block3_conv2', activation='relu'))
    model.add(Conv2D(256, 3, 3, padding='same',
                            name='block3_conv3', activation='relu'))
    model.add(AveragePooling2D(name='block3_pool'))

    # Block 4
    model.add(Conv2D(512, 3, 3, padding='same',
                            name='block4_conv1', activation='relu'))
    model.add(Conv2D(512, 3, 3, padding='same',
                            name='block4_conv2', activation='relu'))
    model.add(Conv2D(512, 3, 3, padding='same',
                            name='block4_conv3', activation='relu'))
    model.add(AveragePooling2D(name='block4_pool'))

    # Block 5
    model.add(Conv2D(512, 3, 3, padding='same',
                            name='block5_conv1', activation='relu'))
    model.add(Conv2D(512, 3, 3, padding='same',
                            name='block5_conv2', activation='relu'))
    model.add(Conv2D(512, 3, 3, padding='same',
                            name='block5_conv3', activation='relu'))
    model.add(AveragePooling2D(name='block5_pool'))

    # Classification block
    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, name='fc1', activation='relu'))
    model.add(Dense(4096, name='fc2', activation='relu'))
    model.add(Dense(1000, name='predictions', activation='softmax'))

    return model


if __name__ == '__main__':

    import json

    class_idx_path = '/home/rbodo/.snntoolbox/Datasets/imagenet/' \
                     'imagenet_class_index_dir.json'
    train_path = '/home/rbodo/.snntoolbox/Datasets/imagenet/training'
    test_path = '/home/rbodo/.snntoolbox/Datasets/imagenet/validation'
    weights_path = '/home/rbodo/.snntoolbox/data/imagenet/vgg16_averagepool/' \
                   '63.40.h5'

    # Build network
    print("Creating pre-trained model...")
    vgg16 = get_vgg16()
    sgd = SGD(0.00001, 0.9, nesterov=True)
    vgg16.compile(sgd, 'categorical_crossentropy', ['accuracy'])

    # Load initial weights
    vgg16.load_weights(weights_path)

    # Get dataset
    print("Loading dataset...")
    class_idx = json.load(open(class_idx_path, "r"))
    classes = []
    for idx in range(len(class_idx)):
        classes.append(class_idx[str(idx)][0])

    target_size = (224, 224)
    batch_size = 32
    samples_per_epoch = 10000
    nb_val_samples = 10000
    nb_epoch = 3

    datagen = ImageDataGenerator()
    trainflow = datagen.flow_from_directory(
        train_path, target_size, classes=classes, batch_size=batch_size)

    testflow = datagen.flow_from_directory(
        test_path, target_size, classes=classes, batch_size=nb_val_samples)

    print("Evaluating pre-trained model...")
    x_eval, y_eval = testflow.next()
    x_eval = preprocess_input(x_eval)
    loss, acc = vgg16.evaluate(x_eval, y_eval, verbose=0)
    print("Val. loss: {}, accuracy: {}".format(loss, acc))

    print("Refining pre-trained model...")
    for e in range(nb_epoch):
        print("Epoch {}".format(e))
        batch_idx = loss = acc = 0
        for x_batch, y_batch in trainflow:
            x_batch = preprocess_input(x_batch)
            loss, acc = vgg16.train_on_batch(x_batch, y_batch)
            print("Training loss: {}, accuracy: {}".format(loss, acc))
            batch_idx += 1
            if batch_idx * batch_size >= samples_per_epoch:
                break
        print("Validating after epoch {}...".format(e))
        x_eval, y_eval = testflow.next()
        x_eval = preprocess_input(x_eval)
        loss, acc = vgg16.evaluate(x_eval, y_eval, verbose=0)
        lr = vgg16.optimizer.lr.get_value()
        print("Val. loss: {}, accuracy: {}, learning rate: {}".format(
            loss, acc, lr))
        # Decay learning rate
        # vgg16.optimizer.lr.set_value(lr*0.1)
        # Save network
        vgg16.save('/home/rbodo/.snntoolbox/data/imagenet/vgg16_averagepool/'
                   '{}.h5'.format(acc))
