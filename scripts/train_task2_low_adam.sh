#! /bin/bash

PYTHON=python3

mkdir -p task2_models

for l in basque bengali bulgarian catalan czech danish dutch english estonian faroese finnish french georgian german haida hebrew hindi hungarian icelandic irish italian khaling kurmanji; do for i in `seq 1 1`; do echo $l $i && $PYTHON scripts/attention_train_task2.py all/task2/"$l"-train-low all/task2/"$l"-uncovered-dev 20000 20 0.03 0.01 task2_models/"$l".low."$i".model; done; done
