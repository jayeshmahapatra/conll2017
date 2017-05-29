#! /bin/bash

for l in albanian arabic armenian basque bengali bulgarian catalan czech danish dutch english estonian faroese finnish french georgian german haida hebrew hindi hungarian icelandic irish italian khaling kurmanji; do for q in low; do for i in `seq 1 10`; do
    rm ./task2_models/"$l"."$q"."$i".model
    rm ./task2_models/"$l"."$q"."$i".model.chars.pkl
done;done;done;