from sys import argv
from collections import defaultdict
import random

ALPHA = 0.0

def sample(probs):
    comp = random.random()
    acc = 0
    for p,c in probs:
        acc += p
        if acc >= comp:
            return c
    assert(0)

class LanModel:
    def __init__(self,words):
        self.characters = list(set([c for wf in words for c in wf] + ['#']))
        self.trigram_counts = {(c1,c2,c3):ALPHA 
                               for c1 in self.characters 
                               for c2 in self.characters 
                               for c3 in self.characters}

        for wf in words:
            wf = '#' + wf + '#'
            for c,nc,nnc in zip(wf,wf[1:],wf[2:]+'#'):
                self.trigram_counts[(c,nc,nnc)] += 1

    def resample_letter(self, i, wf):
        assert(0 < i and i < len(wf) - 1)
 
        probs = [[1.0, c] for c in self.characters]

        for k, c in enumerate(self.characters):
            trigram = (wf[i-1],c,wf[i+1])
            if not c in '# ' and self.trigram_counts[trigram] > 0:
                probs[k][0] += self.trigram_counts[trigram]
            elif c != wf[i]:
                probs[k][0] = 0

        tot = sum([x[0] for x in probs])
        probs = sorted([(count/tot,c) for count, c in probs],reverse=1)
        return sample(probs)

    def resample_letters(self, i, j, wf):
        assert(i > 0 and i < len(wf) - 1)
        res = [c for c in wf]
        for k in range(i,j+1):
            res[k] = self.resample_letter(k,res)
        return ''.join(res)

def cs(i,j,str1,str2):
    for k in range(min(len(str1) - i,
                       len(str2) - j)):
        if str1[i+k] != str2[j+k]:
            return k
    return min(len(str1) - i, len(str2) - j) - 1

def lcs(str1,str2):
    csses = {(i,j):cs(i,j,str1,str2) for i in range(len(str1)) 
             for j in range(len(str2))}

    max_v = 0
    max_k = None
    for k,v in csses.items():
        if v >= max_v:
            max_v = v
            max_k = k
    return max_k, max_v

"""
Input: 

data - list of 3-tuples (LEMMA, WF, TAGS) where LEMMA and WF are
strings and TAGS is a tuple of morphological features
e.g. ('N','PL','GEN','POSS3').

factor - How much augmented data to add. The result has size
factor*len(data).

Output: Combined list of augmented training examples and input
training examples.
"""
def augment(data, factor):
    m = LanModel([l[0] for l in data] + [l[1] for l in data])

    aug_data = []

    for f in range(factor - 1):
        for lemma, wf, labels in data:
            stem_starts, stem_len = lcs(wf,lemma)
            new_wf = m.resample_letters(stem_starts[0] + 1, stem_starts[0] + stem_len, '#' + wf + '#')[1:len(wf)+1]
            new_stem = new_wf[stem_starts[0]:stem_starts[0]+stem_len]
            new_lemma = lemma[:stem_starts[1]] + new_stem + lemma[stem_starts[1]+stem_len:]
            aug_data.append((new_lemma,new_wf,labels))
    aug_data += data
    return aug_data
