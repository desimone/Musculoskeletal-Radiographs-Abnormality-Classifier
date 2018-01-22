# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import keras
import numpy as np
from sklearn.metrics import *


class SKLearnMetrics(keras.callbacks.Callback):
    """ SKLearnMetrics computes various classification metrics at the end of a batch.
     Unforunately, doesn't work when used with generators...."""

    def on_train_begin(self, logs={}):
        self.confusion = []
        self.precision = []
        self.recall = []
        self.f1s = []
        self.kappa = []
        self.auc = []

    def on_epoch_end(self, epoch, logs={}):
        score = np.asarray(self.model.predict(self.validation_data[0]))
        predict = np.round(np.asarray(self.model.predict(self.validation_data[0])))
        target = self.validation_data[1]

        self.auc.append(roc_auc_score(target, score))
        self.confusion.append(confusion_matrix(target, predict))
        self.precision.append(precision_score(target, predict))
        self.recall.append(recall_score(target, predict))
        self.f1s.append(f1_score(target, predict))
        self.kappa.append(cohen_kappa_score(target, predict))
        return
