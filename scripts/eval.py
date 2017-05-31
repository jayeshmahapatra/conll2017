from sys import argv

sysdata = [l for l in open(argv[1]).read().split('\n') if l != '']
sysdata = [l.split('\t')[1] for l in sysdata]
golddata = [l for l in open(argv[2]).read().split('\n') if l != '']
goldtags = [l.split('\t')[2] for l in golddata]
golddata = [l.split('\t')[1] for l in golddata]

assert(len(golddata) == len(sysdata))

knowntags = set()
if len(argv) == 4:
    traindata = [l for l in open(argv[3]).read().split('\n') if l != '']
    knowntags = set([l.split('\t')[2] for l in traindata])

corr = 0
knowncorr = 0
unknowncorr = 0
knowntotal = 0
unknowntotal = 0
for i in range(len(sysdata)):
    corr += (sysdata[i] == golddata[i])
    if sysdata[i] != golddata[i]:
        print(sysdata[i],golddata[i])
    if goldtags[i] in knowntags:
        knowncorr += (sysdata[i] == golddata[i])
        knowntotal += 1
    else:
        unknowncorr += (sysdata[i] == golddata[i])
        unknowntotal += 1
print("ACC: ", corr * 100.0 / len(sysdata))
if knowntags != set():
    if knowntotal > 0:
        print("KNOWN ACC:", knowncorr * 100.0 / knowntotal)
    if unknowntotal > 0:
        print("UNKNOWN ACC:", unknowncorr * 100.0 / unknowntotal)
