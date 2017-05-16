import sys
sys.path.append("/Users/ajwieme/conll2017/baseline")
from baseline import *
from augment import augment

OUTPUT, HELP, PATH = False, False, './../all/'
outpath = "./../post-correction/"

def get_correction_data(data, model):
  correction_data = []

  for index in range(10):
    train, test = train_test_split(data, index)
    correction_data += get_baseline_predictions(model, test)

  return correction_data


def get_augmented_data(lines):
  # DATA AUGMETATION
  aug_in = [(line.split("\t")[0], line.split("\t")[1], tuple(line.split("\t")[2].split(";"))) for line in lines]
  return augment(aug_in, 20)


def train_test_split(data, index):
  data = [tuple(line) for line in data]
  # get size for 90% of data
  test_size = .1 * len(data)
  # split from random into train/test
  start = int(index * test_size)
  end = int(test_size + start)
  test_lines = data[start:end]
  train_lines = list(set(data) - set(test_lines))

  return (train_lines, test_lines)


def get_baseline_model(train_lines):
  allprules, allsrules = {}, {}
  # First, test if language is predominantly suffixing or prefixing
  # If prefixing, work with reversed strings
  prefbias, suffbias = 0, 0
  for l in train_lines:
    lemma, form, _ = l
    aligned = halign(lemma, form)
    if ' ' not in aligned[0] and ' ' not in aligned[1] and '-' not in aligned[0] and '-' not in aligned[1]:
      prefbias += numleadingsyms(aligned[0], '_') + numleadingsyms(aligned[1], '_')
      suffbias += numtrailingsyms(aligned[0], '_') + numtrailingsyms(aligned[1], '_')


  for l in train_lines:  # Read in lines and extract transformation rules from pairs
    lemma, form, msd = l
    if prefbias > suffbias:
      lemma = lemma[::-1]
      form = form[::-1]
    prules, srules = prefix_suffix_rules_get(lemma, form)

    if msd not in allprules and len(prules) > 0:
      allprules[msd] = {}
    if msd not in allsrules and len(srules) > 0:
      allsrules[msd] = {}

    for r in prules:
      if (r[0], r[1]) in allprules[msd]:
        allprules[msd][(r[0], r[1])] = allprules[msd][(r[0], r[1])] + 1
      else:
        allprules[msd][(r[0], r[1])] = 1

    for r in srules:
      if (r[0], r[1]) in allsrules[msd]:
        allsrules[msd][(r[0], r[1])] = allsrules[msd][(r[0], r[1])] + 1
      else:
        allsrules[msd][(r[0], r[1])] = 1

  return [prefbias, suffbias, allprules, allsrules]


def get_baseline_predictions(model, test_lines):
  preds = []
  prefbias, suffbias, allprules, allsrules = model

  for l in test_lines:
    lemma, correct, msd = l
    if prefbias > suffbias:
      lemma = lemma[::-1]
    outform = apply_best_rule(lemma, msd, allprules, allsrules)
    if prefbias > suffbias:
      outform = outform[::-1]

    preds.append([outform, correct, msd])

  return preds


def eval(lang, task, quantity, devlines):
  numcorrect = 0
  numguesses = 0
  num_infinitive = 0
  num_no_change = 0

  if task == 1:
    for l in devlines:
      guess, correct, msd_string, = l.split(u'\t')
      if guess == correct:
        numcorrect += 1
      numguesses += 1

  print("BASELINE ACC:" + lang + "[task " + str(task) + "/" + quantity + "]" + ": " + str(str(numcorrect / float(numguesses)))[0:7])

def main():
  for task in [1]:
    for lang in ["english", "finnish"]:
      for quantity in ['low']:

        # Prepare data and baseline model
        lines = [line.strip() for line in
                 codecs.open(PATH + "task" + str(task) + "/" + lang + "-train-" + quantity, "r", encoding="utf-8")]
        train_data = get_augmented_data(lines)
        baseline_model = get_baseline_model(train_data)

        # TRAIN
        # baseline guess  -  correct  -  msd
        correction_data = get_correction_data(train_data, baseline_model)

        train_outfile = codecs.open(outpath + "task" + str(task) + "/" + lang + "-" + "train-" + quantity, "w",
                              encoding="utf-8")

        # Write the post-correction/train file
        for line in correction_data:
          guess, correct, msd_tuple = line
          train_outfile.write(guess + "\t" + correct + "\t" + ';'.join(msd_tuple) + "\n")

        # DEV
        devlines = [line.strip() for line in
                 codecs.open(PATH + "task" + str(task) + "/" + lang + "-dev", "r", encoding="utf-8")]
        # Format devlines for consistancy to match the output of augment()
        formatted_devlines = []
        for l in devlines:
          lemma, wf, tags = l.split("\t")
          formatted_devlines.append([lemma, wf, tuple(tags.split(";"))])

        # Get the baselne for dev with the augmented train data and model
        dev_data = get_baseline_predictions(baseline_model, formatted_devlines)

        # Write outfile of dev data with guess  -  correct  -  msd
        # so that we can check attention against corrections
        dev_outfile = codecs.open(outpath + "task" + str(task) + "/" + lang + "-dev-" + quantity, "w",
                              encoding="utf-8")

        dev_validation = []
        for line in dev_data:
          guess, correct, msd_tuple = line
          dev_validation.append(guess + "\t" + correct + "\t" + ';'.join(msd_tuple))
          dev_outfile.write(guess + "\t" + correct + "\t" + ';'.join(msd_tuple) + "\n")

        eval(lang, task, quantity, dev_validation)

        # TEST
        testlines = [line.strip() for line in
                    codecs.open(PATH + "task" + str(task) + "/" + lang + "-test", "r", encoding="utf-8")]
        # Format testlines for consistancy to match the output of augment()
        formatted_testlines = []
        for l in testlines:
          lemma, tags = l.split("\t")
          # Add ? as correct palceholder
          formatted_testlines.append([lemma, "?", tuple(tags.split(";"))])

        # Get the baselne for test with the augmented train data
        test_data = get_baseline_predictions(baseline_model, formatted_testlines)

        # Write outfile of dev data with guess  -  correct  -  msd
        # so that we can check attention against corrections
        test_outfile = codecs.open(outpath + "task" + str(task) + "/" + lang + "-test-" + quantity, "w",
                                  encoding="utf-8")
        for line in test_data:
          guess, _, msd_tuple = line
          test_outfile.write(guess + "\t" + ';'.join(msd_tuple) + "\n")
