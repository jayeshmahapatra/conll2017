#! /bin/bash

PYTHON=python3

for l in albanian arabic armenian basque bengali bulgarian catalan czech danish dutch english estonian faroese finnish french georgian german haida hebrew hindi hungarian icelandic irish italian khaling kurmanji latin latvian lithuanian lower-sorbian macedonian navajo northern-sami norwegian-bokmal norwegian-nynorsk persian polish portuguese quechua romanian russian scottish-gaelic serbo-croatian slovak slovene sorani spanish swedish turkish ukrainian urdu welsh; do for q in medium high; do for i in `seq 1 1`; do
    $PYTHON scripts/attention_test_task2.py all/task2/"$l"-covered-dev task2_models/"$l"."$q"."$i".model task2_results/"$l"-test."$q"."$i".sys;
done; done; done
