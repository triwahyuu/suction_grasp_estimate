#!/usr/bin/env bash

set -o xtrace

python3 train.py -a pspeffnetb2 --max-epoch 120 --resume result/pspeffnetb2/20191219_065756/checkpoint.pth.tar
python3 train.py -a resnet34 --max-epoch 120 --resume result/resnet34/20191219_121356/checkpoint.pth.tar
python3 train.py -a bisenet34 --max-epoch 100
python3 train.py -a biseeffnetb0 --max-epoch 100
python3 train.py -a biseeffnetb1 --max-epoch 100
python3 train.py -a biseeffnetb2 --max-epoch 100
python3 train.py -a biseeffnetb3 --max-epoch 100