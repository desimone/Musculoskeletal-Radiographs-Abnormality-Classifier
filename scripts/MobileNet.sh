#!/bin/bash
python train.py \
	--epochs 5 \
	--b 32

python train.py \
	--b 32 \
	--resume ./models/MobileNet.hdf5 \
	--lr 1e-4

python train.py \
	--b 32 \
	--resume ./models/MobileNet.hdf5 \
	--lr 1e-5

python train.py \
	--b 32 \
	--resume ./models/MobileNet.hdf5 \
	--lr 1e-6
