#! /bin/bash

PYTHON=python3

for l in albanian arabic armenian basque bengali bulgarian catalan czech danish dutch english estonian faroese finnish french georgian german haida hebrew hindi hungarian icelandic irish italian khaling kurmanji latin latvian lithuanian lower-sorbian macedonian navajo northern-sami norwegian-bokmal norwegian-nynorsk persian polish portuguese quechua romanian russian scottish-gaelic serbo-croatian slovak slovene sorani spanish swedish turkish ukrainian urdu welsh; do for q in low medium high;
do
    $PYTHON scripts/vote_with_weights.py task1_test_results/"$l"-covered-test."$q" task1_vote_weights/"$l"."$q".sys.weights > task1_test_results/"$l"-covered-test."$q".sys
done; done
