#! /bin/bash

PYTHON=python3

mkdir -p task2_models

for l in slovak slovene sorani spanish swedish turkish ukrainian urdu welsh; do for i in `seq 1 1`; do echo $l $i && $PYTHON scripts/attention_train_task2.py all/task2/"$l"-train-high all/task2/"$l"-uncovered-dev 25000 20 0.03 0.01 task2_models/"$l".high."$i".model; done; done