from sys import argv

sysdata = [l for l in open(argv[1]).read().split('\n') if l != '']
sysdata = [l.split('\t')[1] for l in sysdata]
golddata = [l for l in open(argv[2]).read().split('\n') if l != '']
golddata = [l.split('\t')[1] for l in golddata]
assert(len(golddata) == len(sysdata))

corr = 0
for i in range(len(sysdata)):
    corr += (sysdata[i] == golddata[i])
print(corr * 100.0 / len(sysdata))
