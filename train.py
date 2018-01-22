from __future__ import absolute_import, division, print_function

import argparse
from datetime import datetime
from os import environ

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.applications import MobileNet
from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard)
from keras.layers import Dense, GlobalAveragePooling2D
from keras.metrics import binary_accuracy, binary_crossentropy
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight

environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Shut up tensorflow!
print("tf : {}".format(tf.__version__))
print("keras : {}".format(keras.__version__))
print("numpy : {}".format(np.__version__))
print("pandas : {}".format(pd.__version__))

parser = argparse.ArgumentParser(description='Hyperparameters')
parser.add_argument('--classes', default=1, type=int)
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--epochs', default=120, type=int)
parser.add_argument('-b', '--batch-size', default=32, type=int, help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float)
parser.add_argument('--lr-wait', default=10, type=int, help='how long to wait on plateu')
parser.add_argument('--decay', default=1e-4, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint')
parser.add_argument('--fullretrain', dest='fullretrain', action='store_true', help='retrain all layers of the model')
parser.add_argument('--seed', default=1337, type=int, help='random seed')
parser.add_argument('--img_channels', default=3, type=int)
parser.add_argument('--img_size', default=224, type=int)
parser.add_argument('--early_stop', default=20, type=int)


def train():
    global args
    args = parser.parse_args()
    img_shape = (args.img_size, args.img_size, args.img_channels)  # blame theano
    now_iso = datetime.now().strftime('%Y-%m-%dT%H:%M:%S%z')

    # We then scale the variable-sized images to 224x224
    # We augment .. by applying random lateral inversions and rotations.
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=45,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(
        'data/train',
        shuffle=True,
        target_size=(args.img_size, args.img_size),
        class_mode='binary',
        batch_size=args.batch_size, )

    val_datagen = ImageDataGenerator(rescale=1. / 255)
    val_generator = val_datagen.flow_from_directory(
        'data/val',
        shuffle=True,  # otherwise we get distorted batch-wise metrics
        class_mode='binary',
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch_size, )

    classes = len(train_generator.class_indices)
    assert classes > 0
    assert classes is len(val_generator.class_indices)
    n_of_train_samples = train_generator.samples
    n_of_val_samples = val_generator.samples

    # Architectures
    base_model = MobileNet(input_shape=img_shape, weights='imagenet', include_top=False)
    x = base_model.output  # Recast classification layer
    # x = Flatten()(x)  # Uncomment for Resnet based models
    x = GlobalAveragePooling2D(name='predictions_avg_pool')(x)  # comment for RESNET models
    # n_classes; softmax for multi-class, sigmoid for binary
    x = Dense(args.classes, activation='sigmoid', name='predictions')(x)
    model = Model(inputs=base_model.input, outputs=x)

    # checkpoints
    #
    checkpoint = ModelCheckpoint(filepath='./models/MobileNet.hdf5', verbose=1, save_best_only=True)
    early_stop = EarlyStopping(patience=args.early_stop)
    tensorboard = TensorBoard(log_dir='./logs/MobileNet/{}/'.format(now_iso))
    # reduce_lr = ReduceLROnPlateau(factor=0.03, cooldown=0, patience=args.lr_wait, min_lr=0.1e-6)
    callbacks = [checkpoint, tensorboard, checkpoint]

    # Calculate class weights
    weights = class_weight.compute_class_weight('balanced', np.unique(train_generator.classes), train_generator.classes)
    weights = {0: weights[0], 1: weights[1]}
    # for layer in base_model.layers:
    #     layer.set_trainable = False

    # print(model.summary())
    # for i, layer in enumerate(base_model.layers):
    #     print(i, layer.name)
    if args.resume:
        model.load_weights(args.resume)
        for layer in model.layers:
            layer.set_trainable = True

    # if TRAIN_FULL:
    #     print("=> retrain all layers of network")
    #     for layer in model.layers:
    #         set_trainable = True
    # else:
    #     print("=> retraining only bottleneck and fc layers")
    #     import pdb
    #     pdb.set_trace()
    #     set_trainable = False
    #     for layer in base_model.layers:
    #         if "block12" in layer.name:  # what block do we want to start unfreezing
    #             set_trainable = True
    #         if set_trainable:
    #             layer.trainable = True
    #         else:
    #             layer.trainable = False

    # The network is trained end-to-end using Adam with default parameters
    model.compile(
        optimizer=Adam(lr=args.lr, decay=args.decay),
        # optimizer=SGD(lr=args.lr, decay=args.decay,momentum=args.momentum, nesterov=True),
        loss=binary_crossentropy,
        metrics=[binary_accuracy], )

    model_out = model.fit_generator(
        train_generator,
        steps_per_epoch=n_of_train_samples // args.batch_size,
        epochs=args.epochs,
        validation_data=val_generator,
        validation_steps=n_of_val_samples // args.batch_size,
        class_weight=weights,
        workers=args.workers,
        use_multiprocessing=True,
        callbacks=callbacks)


if __name__ == '__main__':
    train()
