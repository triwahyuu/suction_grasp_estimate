#!/usr/bin/env bash

set -o xtrace

## all models
# resnet=$(echo resnet{18,34,50,101})
# pspnet=$(echo pspnet{18,34,50,101})
# bisenet=$(echo bisenet{34,18,50})
# fcneffnet=$(echo fcneffnetb{0..4})
# pspeffnet=$(echo pspeffnetb{0..4})
# biseeffnet=$(echo biseeffnetb{0..4})

resnet=$(echo resnet{18,50,101})
pspnet=$(echo pspnet{18,101})
bisenet=$(echo bisenet{34,18,50})
fcneffnet=$(echo fcneffnetb{0,4})
pspeffnet=$(echo pspeffnetb{0,4,1,3})
biseeffnet=$(echo biseeffnetb{0..4})

function train(){
    for a in $@; do
        python3 train.py -a $a --max-epoch 100
    done
}

function train2(){
    for a in $@; do
        python3 train.py -a $a --max-epoch 120
    done
}

# train $bisenet $biseeffnet $resnet $pspnet $fcneffnet $pspeffnet 
# train $pspeffnet $resnet $pspnet $fcneffnet
train $pspeffnet
train2 $biseeffnet $bisenet