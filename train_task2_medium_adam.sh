#! /bin/bash

PYTHON=python3

mkdir -p task2_models

for l in albanian arabic armenian basque bengali bulgarian catalan czech danish dutch english estonian faroese finnish french georgian german haida hebrew hindi hungarian icelandic irish italian khaling kurmanji; do for i in `seq 1 5`; do echo $l $i && $PYTHON scripts/attention_train_task2.py all/task2/"$l"-train-medium all/task2/"$l"-uncovered-dev 25000 20 0.03 0.03 task2_models/"$l".medium."$i".model; done; done