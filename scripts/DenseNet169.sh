#!/bin/bash
python train.py \
	--epochs 10 \
	--b 8

python train.py \
	--epochs 15 \
	--b 8 \
	--resume ./models/DenseNet169.hdf5 \
	--lr 1e-4

python train.py \
	--epochs 30 \
	--b 8 \
	--resume ./models/DenseNet169.hdf5 \
	--lr 1e-5

python train.py \
	--epochs 60 \
	--b 8 \
	--resume ./models/DenseNet169.hdf5 \
	--lr 1e-6
