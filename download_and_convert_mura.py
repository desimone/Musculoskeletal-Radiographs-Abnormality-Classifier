# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import re
import os
from os import getcwd
from os.path import exists, isdir, isfile, join
import shutil
import numpy as np
import pandas as pd


class ImageString(object):
    _patient_re = re.compile(r'patient(\d+)')
    _study_re = re.compile(r'study(\d+)')
    _image_re = re.compile(r'image(\d+)')
    _study_type_re = re.compile(r'XR_(\w+)')

    def __init__(self, img_filename):
        self.img_filename = img_filename
        self.patient = self._parse_patient()
        self.study = self._parse_study()
        self.image_num = self._parse_image()
        self.study_type = self._parse_study_type()
        self.image = self._parse_image()
        self.normal = self._parse_normal()

    def flat_file_name(self):
        return "{}_{}_patient{}_study{}_image{}.png".format(self.normal, self.study_type, self.patient, self.study,
                                                            self.image, self.normal)

    def _parse_patient(self):
        return int(self._patient_re.search(self.img_filename).group(1))

    def _parse_study(self):
        return int(self._study_re.search(self.img_filename).group(1))

    def _parse_image(self):
        return int(self._image_re.search(self.img_filename).group(1))

    def _parse_study_type(self):
        return self._study_type_re.search(self.img_filename).group(1)

    def _parse_normal(self):
        return "normal" if ("negative" in self.img_filename) else "abnormal"


# processed
# data
# ├── train
# │   ├── abnormal
# │   └── normal
# └── val
#     ├── abnormal
#     └── normal
proc_data_dir = join(getcwd(), 'data')
proc_train_dir = join(proc_data_dir, 'train')
proc_val_dir = join(proc_data_dir, 'val')

# Data loading code
orig_data_dir = join(getcwd(), 'MURA-v1.0')
train_dir = join(orig_data_dir, 'train')
train_csv = join(orig_data_dir, 'train.csv')
val_dir = join(orig_data_dir, 'valid')
val_csv = join(orig_data_dir, 'valid.csv')
test_dir = join(orig_data_dir, 'test')
assert isdir(orig_data_dir) and isdir(train_dir) and isdir(val_dir) and isdir(test_dir)
assert exists(train_csv) and isfile(train_csv) and exists(val_csv) and isfile(val_csv)

df = pd.read_csv(train_csv, names=['img', 'label'], header=None)
# imgs = df.img.values.tolist()
# labels = df.label.values.tolist()
# following datasets/folder.py's weird convention here...
samples = [tuple(x) for x in df.values]
for img, label in samples:
    assert ("negative" in img) is (label is 0)
    enc = ImageString(img)
    cat_dir = join(proc_train_dir, enc.normal)
    if not os.path.exists(cat_dir):
        os.mkdir(cat_dir)
    shutil.copy2(enc.img_filename, join(cat_dir, enc.flat_file_name()))

df = pd.read_csv(val_csv, names=['img', 'label'], header=None)
samples = [tuple(x) for x in df.values]
for img, label in samples:
    assert ("negative" in img) is (label is 0)
    enc = ImageString(img)
    cat_dir = join(proc_val_dir, enc.normal)
    if not os.path.exists(cat_dir):
        os.mkdir(cat_dir)
    shutil.copy2(enc.img_filename, join(cat_dir, enc.flat_file_name()))
