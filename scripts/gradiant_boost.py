import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction import DictVectorizer
import codecs
from augment import augment

grid_params = {}

model =  GradientBoostingClassifier()
model_cv = GridSearchCV(model, grid_params, scoring="accuracy")

OUTPUT, HELP, PATH = False, False, './../all/'

def extract_features(line):
  lemma, _, msd = line
  ret_dict = {}
  for tag in msd:
    ret_dict["tag=%s" % tag] = 1
  ret_dict["msd=%s" % ";".join(msd)] = 1
  ret_dict["lemma=%s" % lemma] = 1
  for bigram in zip(lemma, lemma[1:]):
    ret_dict["bigram=%s%s" % bigram] = 1

  return ret_dict

def extract_correct(line):
  _, correct, _ = line
  return correct

for task in [1]:
  runningavgLow, runningavgMed, runningavgHigh, numiterLow, numiterMed, numiterHigh = 0.0, 0.0, 0.0, 0, 0, 0
  for lang in sorted(
      set(x.split('-train')[0] for x in os.listdir(PATH + "task" + str(task) + "/") if x.split('-train')[0] in [
        "english"])):
    for quantity in ['low']:
      allprules, allsrules = {}, {}
      if not os.path.isfile(PATH + "task" + str(task) + "/" + lang + "-train-" + quantity):
        continue
      lines = [line.strip() for line in
               codecs.open(PATH + "task" + str(task) + "/" + lang + "-train-" + quantity, "r", encoding="utf-8")]
      # DATA AUGMETATION
      aug_in = [(line.split("\t")[0], line.split("\t")[1], tuple(line.split("\t")[2].split(";"))) for line in lines]
      augmented_data = augment(aug_in, 10)

      # vectorize features
      dv = DictVectorizer(sparse=True)
      features_dicts = [extract_features(line) for line in augmented_data]
      possible_features = [key for d in features_dicts for key in d.keys()]
      sparse_matrix = dv.fit_transform(features_dicts)

      # Set inputs
      X = sparse_matrix
      Y = [extract_correct(line) for line in augmented_data]
      print X
      print Y
      model.fit(X, Y)
      print "fitted!"
      devlines = [line.strip().split("\t") for line in
                  codecs.open(PATH + "task" + str(task) + "/" + lang + "-dev", "r", encoding="utf-8")]
      for i, line in enumerate(devlines):
        devlines[i][2] = tuple(line[2].split(";"))

      correct = [(line[1], line[2]) for line in devlines]
      dev_features = [extract_features(line) for line in devlines]
      dev_X = dv.transform(dev_features)
      preds = model.predict(dev_X.toarray())
      compare = zip(preds, correct)
      corr = 0
      for p, c in compare:
        wf, tag = c
        print (p, wf, tag)
        if p == wf:
          corr +=1
      print("DEV ACC %.2f\n" % (corr * 100.0 / len(preds)))
'''
task = 1
lang = "english"
devlines = [line.strip().split("\t") for line in
            codecs.open(PATH + "task" + str(task) + "/" + lang + "-dev", "r", encoding="utf-8")]
for i, line in enumerate(devlines):
  devlines[i][2] = tuple(line[2].split(";"))
correct = [(line[1], line[2]) for line in devlines]
dev_features = [extract_features(line) for line in devlines]
print correct
print dev_features
'''
