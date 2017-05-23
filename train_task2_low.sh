#! /bin/bash

PYTHON=python3

mkdir -p task2_models

for l in albanian arabic armenian basque bengali bulgarian catalan czech danish dutch english estonian faroese finnish french georgian german haida hebrew hindi hungarian icelandic irish italian khaling kurmanji latin latvian lithuanian lower-sorbian macedonian navajo northern-sami norwegian-bokmal norwegian-nynorsk persian polish portuguese quechua romanian russian scottish-gaelic serbo-croatian slovak slovene sorani spanish swedish turkish ukrainian urdu welsh; do for i in `seq 1 10`; do $PYTHON scripts/attention_train.py all/task2/"$l"-train-low all/task2/"$l"-uncovered-dev 50 20 0.03 0.01 task2_models/"$l".low."$i".model; done; done