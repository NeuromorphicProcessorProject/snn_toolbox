import os
import json

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.callbacks import TensorBoard
from keras.metrics import top_k_categorical_accuracy
from keras.optimizers import SGD

data_path = '/home/rbodo/.snntoolbox/Datasets/imagenet'
train_path = os.path.join(data_path, 'training')
test_path = os.path.join(data_path, 'validation')
class_idx_path = os.path.join(data_path, 'imagenet_class_index_1000.json')
log_path = '/home/rbodo/.snntoolbox/data/imagenet/vgg16_trained2'

num_classes = 100

print("Instantiating model...")
model = VGG16(weights=None, classes=num_classes)

sgd = SGD(0.01)
model.compile(sgd, 'categorical_crossentropy',
              ['accuracy'])  # , top_k_categorical_accuracy])

# Get dataset
print("Loading dataset...")
class_idx = json.load(open(class_idx_path, "r"))
classes = [class_idx[str(idx)][0] for idx in range(len(class_idx))]
classes = classes[:num_classes]

target_size = (224, 224)
batch_size = 1
nb_train_samples = 129395  # 1281167
nb_train_steps = nb_train_samples / batch_size
nb_val_samples = 50000  # 50000
nb_val_steps = nb_val_samples / batch_size
nb_epoch = 10

datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
trainflow = datagen.flow_from_directory(
    train_path, target_size, classes=classes, batch_size=batch_size)
testflow = datagen.flow_from_directory(
    test_path, target_size, classes=classes, batch_size=nb_val_samples)
testdata = testflow.next()

# print("Evaluating initial model...")
# score = model.evaluate_generator(testflow, nb_val_steps)
# print("Validation accuracy: {} top-1, {} top-5".format(score[1], score[2]))
# 0.15% and 0.54%

print("Training...")
gradients = TensorBoard(log_path + '/logs_32bit', 1, write_grads=True)
model.fit_generator(trainflow, nb_train_steps, nb_epoch, verbose=1,
                    validation_data=testdata, callbacks=[gradients])

model.save(os.path.join(log_path, 'vgg16.h5'))
