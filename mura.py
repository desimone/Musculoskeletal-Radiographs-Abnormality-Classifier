from __future__ import absolute_import, division, print_function

import re

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score, precision_score, recall_score)

pd.set_option('display.max_rows', 20)
pd.set_option('precision', 4)
np.set_printoptions(precision=4)


class Mura(object):
    """`MURA <https://stanfordmlgroup.github.io/projects/mura/>`_ Dataset :
    Towards Radiologist-Level Abnormality Detection in Musculoskeletal Radiographs.
    """
    url = "https://cs.stanford.edu/group/mlgroup/mura-v1.0.zip"
    filename = "mura-v1.0.zip"
    md5_checksum = '4c36feddb7f5698c8bf291b912c438b1'
    _patient_re = re.compile(r'patient(\d+)')
    _study_re = re.compile(r'study(\d+)')
    _image_re = re.compile(r'image(\d+)')
    _study_type_re = re.compile(r'_(\w+)_patient')

    def __init__(self, image_file_names, y_true, y_pred=None):
        self.imgs = image_file_names
        df_img = pd.Series(np.array(image_file_names), name='img')
        self.y_true = y_true
        df_true = pd.Series(np.array(y_true), name='y_true')
        self.y_pred = y_pred
        # number of unique classes
        self.patient = []
        self.study = []
        self.study_type = []
        self.image_num = []
        self.encounter = []
        for img in image_file_names:
            self.patient.append(self._parse_patient(img))
            self.study.append(self._parse_study(img))
            self.image_num.append(self._parse_image(img))
            self.study_type.append(self._parse_study_type(img))
            self.encounter.append("{}_{}_{}".format(
                self._parse_study_type(img),
                self._parse_patient(img),
                self._parse_study(img), ))

        self.classes = np.unique(self.y_true)
        df_patient = pd.Series(np.array(self.patient), name='patient')
        df_study = pd.Series(np.array(self.study), name='study')
        df_image_num = pd.Series(np.array(self.image_num), name='image_num')
        df_study_type = pd.Series(np.array(self.study_type), name='study_type')
        df_encounter = pd.Series(np.array(self.encounter), name='encounter')

        self.data = pd.concat(
            [
                df_img,
                df_encounter,
                df_true,
                df_patient,
                df_patient,
                df_study,
                df_image_num,
                df_study_type,
            ], axis=1)

        if self.y_pred is not None:
            self.y_pred_probability = self.y_pred.flatten()
            self.y_pred = self.y_pred_probability.round().astype(int)
            df_y_pred = pd.Series(self.y_pred, name='y_pred')
            df_y_pred_probability = pd.Series(self.y_pred_probability, name='y_pred_probs')
            self.data = pd.concat((self.data, df_y_pred, df_y_pred_probability), axis=1)

    def __len__(self):
        return len(self.imgs)

    def _parse_patient(self, img_filename):
        return int(self._patient_re.search(img_filename).group(1))

    def _parse_study(self, img_filename):
        return int(self._study_re.search(img_filename).group(1))

    def _parse_image(self, img_filename):
        return int(self._image_re.search(img_filename).group(1))

    def _parse_study_type(self, img_filename):
        return self._study_type_re.search(img_filename).group(1)

    def metrics(self):
        return "per image metrics:\n\taccuracy : {:.2f}\tf1 : {:.2f}\tprecision : {:.2f}\trecall : {:.2f}\tcohen_kappa : {:.2f}".format(
            accuracy_score(self.y_true, self.y_pred),
            f1_score(self.y_true, self.y_pred),
            precision_score(self.y_true, self.y_pred),
            recall_score(self.y_true, self.y_pred),
            cohen_kappa_score(self.y_true, self.y_pred), )

    def metrics_by_encounter(self):
        y_pred = self.data.groupby(['encounter'])['y_pred_probs'].mean().round()
        y_true = self.data.groupby(['encounter'])['y_true'].mean().round()
        return "per encounter metrics:\n\taccuracy : {:.2f}\tf1 : {:.2f}\tprecision : {:.2f}\trecall : {:.2f}\tcohen_kappa : {:.2f}".format(
            accuracy_score(y_true, y_pred),
            f1_score(y_true, y_pred),
            precision_score(y_true, y_pred),
            recall_score(y_true, y_pred),
            cohen_kappa_score(self.y_true, self.y_pred), )

    # def metrics_by_study_type(self):
    #     y_pred = self.data.groupby(['study_type', 'encounter'])['y_pred_probs'].mean().round()
    #     y_true = self.data.groupby(['study_type', 'encounter'])['y_true'].mean().round()
    #     return "per study_type metrics:\n\taccuracy : {:.2f}\tf1 : {:.2f}\tprecision : {:.2f}\trecall : {:.2f}".format(
    #         accuracy_score(y_true, y_pred),
    #         f1_score(y_true, y_pred),
    #         precision_score(y_true, y_pred),
    #         recall_score(y_true, y_pred), )
