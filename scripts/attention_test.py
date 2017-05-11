from __future__ import print_function
from sys import argv, stderr
import random
import pickle

import dynet as dy

from augment import augment
import attention

if __name__=='__main__':
    if len(argv) != 4:
        stderr.write("USAGE: python3 %s test_file model_file output_file ", argv[0])
        exit(1)

    TEST_FN = argv[1]
    MODEL_FN = argv[2]
    O_FN = argv[3]

    testdata = [l.strip().split('\t') for l in open(TEST_FN).read().split('\n') if l.strip() != '']
    itestdata = [([c for c in lemma],tags.split(';')) for lemma, tags in testdata]

    attention.init_models(None,MODEL_FN)
    attention.load_model(MODEL_FN)
    attention.test(itestdata, open(O_FN,"w"))

