#!/usr/bin/env bash

for file in result/best_models/*
do
    echo 'evaluating '$file
    python evaluate.py --checkpoint $file
done