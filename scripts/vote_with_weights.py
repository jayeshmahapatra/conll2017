from sys import argv, stderr
from collections import defaultdict as dd


def vote(i, data, n, weights):
  counts = dd(float)
  for j in range(n):
    counts[data[j][i]] += weights[j]
  return sorted([(v, k) for k, v in counts.items()], reverse=1)[0][1]


if __name__ == '__main__':
  if len(argv) < 3:
    stderr.write("USAGE: %s prefix [N]")
    exit(1)

  N = 10
  if len(argv) == 4:
    N = int(argv[3])

  weights_FN = argv[2]

  weights = dd(lambda: 0.0)
  weights_list = open(argv[2]).read().split("\t")
  for i, weight in enumerate(weights_list):
    weights[i] = float(weight)

  print(weights)

  data_sets = []
  for i in range(1, N + 1):
    data_sets.append(open("%s.%u.sys" % (argv[1], i)).read().split('\n'))
    data_sets[-1] = [wf for wf in data_sets[-1] if wf != '']

  for i in range(len(data_sets[0])):
    print(vote(i, data_sets, N, weights))
