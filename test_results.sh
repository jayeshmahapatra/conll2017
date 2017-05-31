#! /bin/bash

PYTHON=python3

for l in albanian; do for q in low high; do for i in `seq 1 1`; do
    $PYTHON scripts/attention_test_task2.py all/task2/"$l"-covered-dev task2_models/"$l"."$q"."$i".model task2_results/"$l"-test."$q"."$i".sys;
done; done; done
