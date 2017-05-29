#! /bin/bash

PYTHON=python3

for l in albanian arabic armenian basque bengali bulgarian catalan czech danish dutch english estonian faroese finnish french georgian german haida hebrew hindi hungarian icelandic irish italian khaling kurmanji; do for i in `seq 1 1`; do
    $PYTHON scripts/attention_test_task2.py all/task2/"$l"-covered-dev task2_models/"$l".medium."$i".model task2_results/"$l"-test.medium."$i".sys;
done; done
