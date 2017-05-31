#! /bin/bash

PYTHON=python3

mkdir -p task2_models

for l in latvian lithuanian lower-sorbian macedonian navajo northern-sami norwegian-bokmal norwegian-nynorsk persian polish portuguese quechua romanian russian scottish-gaelic serbo-croatian slovak slovene sorani spanish swedish turkish ukrainian urdu welsh; do for i in `seq 1 1`; do echo $l $i && $PYTHON scripts/attention_train_task2.py all/task2/"$l"-train-medium all/task2/"$l"-uncovered-dev 25000 20 0.03 0.03 task2_models/"$l".medium."$i".model; done; done