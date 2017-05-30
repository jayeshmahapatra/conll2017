#! /bin/bash

PYTHON=python3

for l in slovak slovene sorani spanish swedish turkish ukrainian urdu welsh; do for i in `seq 1 1`; do
    $PYTHON scripts/attention_test_task2.py all/task2/"$l"-covered-dev task2_models/"$l".high."$i".model task2_results/"$l"-test.high."$i".sys;
done; done
