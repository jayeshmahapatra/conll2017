from sys import argv, stderr
from collections import defaultdict as dd

def vote(i,data):
    counts = dd(lambda : 0)
    for j in range(10):
        counts[data[j][i]] += 1
    return sorted([(v,k) for k,v in counts.items()],reverse=1)[0][1]

if __name__=='__main__':
    if len(argv) != 2:
        stderr.write("USAGE: %s prefix")
        exit(1)
    data_sets = []
    for i in range(1,11):
        data_sets.append(open("%s.%u.sys" % (argv[1],i)).read().split('\n'))
        data_sets[-1] = [wf for wf in data_sets[-1] if wf != '']
    
    for i in range(len(data_sets[0])):
        print(vote(i,data_sets))
