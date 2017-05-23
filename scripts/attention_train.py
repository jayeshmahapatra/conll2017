from __future__ import print_function
from sys import argv, stderr
import random
import pickle

import dynet as dy

from augment import augment
import attention

if __name__=='__main__':
    if len(argv) != 8:
        stderr.write(("USAGE: python3 %s train_file dev_file "+
                      "aug_factor num_epochs alpha beta modeloutfile\n") % argv[0])
        exit(1)

    TRAIN_FN=argv[1]
    DEV_FN=argv[2]
    AUG_FACTOR=int(argv[3])
    attention.EPOCHS=int(argv[4])
    ALPHA=float(argv[5])
    BETA=float(argv[6])
    O_FN = argv[7]

    data = augment([l.strip().split('\t') for l in open(TRAIN_FN).read().split('\n') if l.strip() != ''],AUG_FACTOR)

    
    idata = [[c for c in lemma] + tags.split(';') for lemma, _, tags in data]
    odata = [[c for c in wf] for _, wf, _ in data]    
    devdata = [l.strip().split('\t') for l in open(DEV_FN).read().split('\n') if l.strip() != '']
    idevdata = [[c for c in lemma] + tags.split(';') for lemma, _, tags in devdata]
    odevdata = [[c for c in wf] for _, wf, _ in devdata]
    
    characters = set([attention.EOS])
    
    for wf in idata + odata + idevdata + odevdata:
        for c in wf:
            characters.add(c)

    attention.init_models(characters, None)
    attention.train(idata,odata,idevdata,odevdata,ALPHA, BETA, O_FN)


