from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import time
import sys
import json

# Example usage: python SotA_methods.py <max length of n-grams extracted>
# By default set to 2-grams


def output(method):
    """This method makes predictions for the test set and prints the results"""
    startT = time.time()
    yhat = regr.predict(testX)
    endT = time.time()
    predictTime = endT - startT
    outfile_name = method + str(max_ngram) + '.conc.pred'
    np.savetxt(outfile_name, yhat)

    # Print Stats
    with open(method + str(max_ngram) + '.stats.json', 'w') as outfile:
        json.dump(
            [{"method": method,
              "max_ngram": max_ngram,
              "MSE": np.mean((regr.predict(testX) - testy)**2),
              "MAE": np.mean(np.abs(regr.predict(testX) - testy)),
              "score": regr.score(testX, testy),
              "toppos_feature": vectorizer.get_feature_names()[
                  np.argmax(regr.coef_)],
              "toppos_feature_weigth": np.max(regr.coef_),
              "topneg_feature": vectorizer.get_feature_names()[
                  np.argmin(regr.coef_)],
              "topneg_feature_weight": np.min(regr.coef_),
              "read_training_time": readTrainingTime,
              "read_test_time": readTestTime,
              "extract_training_time": extractTrainingTime,
              "extract_test_time": extractTestTime,
              "learning_time": learnTime,
              "predict_test_time": predictTime}],
            outfile)


trainfile = sys.argv[2]
testfile = sys.argv[3]
# Load training and test files
startT = time.time()
# trainfile = "/home/svgsponer/svgphd/tmp/toydata/toySequence.train"
seqs = np.genfromtxt(trainfile, delimiter=" ", dtype=None, usecols=(1))
y = np.genfromtxt(trainfile, delimiter=" ", dtype=None, usecols=(0))
endT = time.time()
readTrainingTime = endT - startT

startT = time.time()
testfile = "/home/svgsponer/svgphd/tmp/toydata/toySequence.test"
testseqs = np.genfromtxt(testfile, delimiter=" ", dtype=None, usecols=(1))
testy = np.genfromtxt(testfile, delimiter=" ", dtype=None, usecols=(0))
endT = time.time()
readTestTime = endT - startT

# Extract n-grams
if len(sys.argv) < 2:
    max_ngram = 2
else:
    max_ngram = int(sys.argv[1])
vectorizer = CountVectorizer(
    min_df=1, binary=True, analyzer='char', ngram_range=(1, max_ngram))
startT = time.time()
X = vectorizer.fit_transform(seqs)
endT = time.time()
extractTrainingTime = endT - startT

X.shape
startT = time.time()
testX = vectorizer.transform(testseqs)
endT = time.time()
extractTestTime = endT - startT

# Learn model (Linear regression)
print('Start OLS')
from sklearn import linear_model
regr = linear_model.LinearRegression()
startT = time.time()
regr.fit(X, y)
endT = time.time()
learnTime = endT - startT
output('ols')
print()

# Ridge model (Rigde Regression)
print('Start Ridge')
from sklearn import linear_model
regr = linear_model.Ridge()
startT = time.time()
regr.fit(X, y)
endT = time.time()
learnTime = endT - startT
output('ridge')
print()

regr.coef_
regr.score(testX, testy)

# Learn model (ElaticNet())
print('Start Enet')
from sklearn import linear_model
regr = linear_model.ElasticNet()
startT = time.time()
regr.fit(X, y)
endT = time.time()
learnTime = endT - startT
output('Enet')
print()

# Learn model (linearSVR)
print('Start SVR')
from sklearn import svm
regr = svm.LinearSVR()
startT = time.time()
regr.fit(X, y)
endT = time.time()
learnTime = endT - startT
output('linsvr')
print()

# Learn model (Lasso())
# from sklearn import linear_model
# regr = linear_model.Lasso()
# regr.fit(X, y)
