from __future__ import absolute_import, division, print_function

import re
from os import getcwd
from os.path import join

import numpy as np
import pandas as pd
import torch.utils.data as data
from PIL import Image

from torchvision.datasets.utils import check_integrity, download_url


class MuraDataset(data.Dataset):
    """`MURA <https://stanfordmlgroup.github.io/projects/mura/>`_ Dataset :
    Towards Radiologist-Level Abnormality Detection in Musculoskeletal Radiographs.
    """
    url = "https://cs.stanford.edu/group/mlgroup/mura-v1.0.zip"
    filename = "mura-v1.0.zip"
    md5_checksum = '4c36feddb7f5698c8bf291b912c438b1'
    _patient_re = re.compile(r'patient(\d+)')
    _study_re = re.compile(r'study(\d+)')
    _image_re = re.compile(r'image(\d+)')
    _study_type_re = re.compile(r'XR_(\w+)')

    def __init__(self, csv_f, transform=None, download=False):
        self.df = pd.read_csv(csv_f, names=['img', 'label'], header=None)
        self.imgs = self.df.img.values.tolist()
        self.labels = self.df.label.values.tolist()
        # following datasets/folder.py's weird convention here...
        self.samples = [tuple(x) for x in self.df.values]
        # number of unique classes
        self.classes = np.unique(self.labels)
        self.balanced_weights = self.balance_class_weights()

        self.transform = transform

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

    def download_and_uncompress_tarball(tarball_url, dataset_dir):
        """Downloads the `tarball_url` and uncompresses it locally.
        Args:
            tarball_url: The URL of a tarball file.
            dataset_dir: The directory where the temporary files are stored.
        """
        filename = tarball_url.split('/')[-1]
        filepath = os.path.join(dataset_dir, filename)

        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                             float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(tarball_url, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        if ".zip" in filename:
            print("zipfile:{}".format(filepath))
            with zipfile.ZipFile(filepath, "r") as zip_ref:
                zip_ref.extractall(dataset_dir)
        else:
            tarfile.open(filepath, 'r:gz').extractall(dataset_dir)

    def balance_class_weights(self):
        count = [0] * len(self.classes)
        for item in self.samples:
            count[item[1]] += 1
        weight_per_class = [0.] * len(self.classes)
        N = float(sum(count))
        for i in range(len(self.classes)):
            weight_per_class[i] = N / float(count[i])
        weight = [0] * len(self.samples)
        for idx, val in enumerate(self.samples):
            weight[idx] = weight_per_class[val[1]]
        return weight

    def __getitem__(self, idx):
        img_filename = join(self.imgs[idx])
        patient = self._parse_patient(img_filename)
        study = self._parse_study(img_filename)
        image_num = self._parse_image(img_filename)
        study_type = self._parse_study_type(img_filename)

        # todo(bdd) : inconsistent right now, need param for grayscale / RGB
        # todo(bdd) : 'L' -> gray, 'RGB' -> Colors
        image = Image.open(img_filename).convert('RGB')
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)

        meta_data = {
            'y_true': label,
            'img_filename': img_filename,
            'patient': patient,
            'study': study,
            'study_type': study_type,
            'image_num': image_num,
            'encounter': "{}_{}_{}".format(study_type, patient, study)
        }
        return image, label, meta_data


if __name__ == '__main__':
    import torchvision.transforms as transforms
    import pprint

    data_dir = join(getcwd(), 'MURA-v1.0')
    val_csv = join(data_dir, 'valid.csv')
    val_loader = data.DataLoader(
        MuraDataset(val_csv,
                    transforms.Compose([
                        transforms.Resize(224),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                    ])),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=False)

    for i, (image, label, meta_data) in enumerate(val_loader):
        pprint.pprint(meta_data.cpu())
        if i == 40:
            break
