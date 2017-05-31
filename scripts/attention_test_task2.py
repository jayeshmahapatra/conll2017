from __future__ import print_function
from sys import argv, stderr
import random
import pickle

import dynet as dy

from augment import augment
import attention_task2

if __name__=='__main__':
    if len(argv) != 4:
        stderr.write("USAGE: python3 %s test_file model_file output_file ", argv[0])
        exit(1)

    TEST_FN = argv[1]
    MODEL_FN = argv[2]
    O_FN = argv[3]

    testdata = [l.strip().split('\t') for l in open(TEST_FN).read().split('\n') if l.strip() != '']     
    testdata = [([c for c in x[0]],x[1],x[2].split(';')) for x in testdata]

    attention_task2.EMBEDDINGS_SIZE = 32
    attention_task2.STATE_SIZE = 32
    attention_task2.ATTENTION_SIZE = 32
    attention_task2.init_models(None, MODEL_FN)
    attention_task2.load_model(MODEL_FN)
    attention_task2.test(testdata, open(O_FN, "w"))
