from __future__ import absolute_import, division, print_function

from os import environ, getcwd
from os.path import join

import keras
import numpy as np
import pandas as pd
import sklearn as skl
import tensorflow as tf
from keras.applications import NASNetMobile
from keras.layers import Dense, GlobalAveragePooling2D
from keras.metrics import binary_accuracy, binary_crossentropy
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from mura import Mura

pd.set_option('display.max_rows', 20)
pd.set_option('precision', 4)
np.set_printoptions(precision=4)

environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Shut up tensorflow!
print("tf : {}".format(tf.__version__))
print("keras : {}".format(keras.__version__))
print("numpy : {}".format(np.__version__))
print("pandas : {}".format(pd.__version__))
print("sklearn : {}".format(skl.__version__))

# Hyper-parameters / Globals
BATCH_SIZE = 512  # tweak to your GPUs capacity
IMG_HEIGHT = 224  # ResNetInceptionv2 & Xception like 299, ResNet50/VGG/Inception 224, NASM 331
IMG_WIDTH = IMG_HEIGHT
CHANNELS = 3
DIMS = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)  # blame theano
MODEL_TO_EVAL = './models/NASNetMobile.hdf5'
DATA_DIR = 'MURA-v1.0/'
EVAL_CSV = 'valid.csv'
EVAL_DIR = 'data/val'

# load up our csv with validation factors
data_dir = join(getcwd(), DATA_DIR)
eval_csv = join(data_dir, EVAL_CSV)
df = pd.read_csv(eval_csv, names=['img', 'label'], header=None)
eval_imgs = df.img.values.tolist()
eval_labels = df.label.values.tolist()

eval_datagen = ImageDataGenerator(rescale=1. / 255)
eval_generator = eval_datagen.flow_from_directory(
    EVAL_DIR, class_mode='binary', shuffle=False, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE)
n_samples = eval_generator.samples
base_model = NASNetMobile(input_shape=DIMS, weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D(name='avg_pool')(x)  # comment for RESNET
x = Dense(1, activation='sigmoid', name='predictions')(x)
model = Model(inputs=base_model.input, outputs=x)
model.load_weights(MODEL_TO_EVAL)
model.compile(optimizer=Adam(lr=1e-3), loss=binary_crossentropy, metrics=['binary_accuracy'])
score, acc = model.evaluate_generator(eval_generator, n_samples / BATCH_SIZE)
print(model.metrics_names)
print('==> Metrics with eval')
print("loss :{:0.4f} \t Accuracy:{:0.4f}".format(score, acc))
y_pred = model.predict_generator(eval_generator, n_samples / BATCH_SIZE)
mura = Mura(eval_generator.filenames, y_true=eval_generator.classes, y_pred=y_pred)
print('==> Metrics with predict')
print(mura.metrics())
print(mura.metrics_by_encounter())
# print(mura.metrics_by_study_type())
