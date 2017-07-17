#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''Creates k stratisfied folds.

Each fold contains: TrainingSet, ValidaitonSet, TestSet
ValidationSet is created by selecting 10% of the TrainingSet.
'''

import pandas as pd
from sklearn.model_selection import StratifiedKFold as KFold
from sklearn.model_selection import train_test_split

filepath = "INSET FILE PATH HERE"
output_path = filepath + "output/"
labels = pd.read_csv(filepath + "trainLabels.csv")


kf = KFold(n_splits=4, shuffle=True)
kf.get_n_splits(labels)


i = 0
for train, test in kf.split(labels['Id'], labels['Class']):
    df_train = labels.iloc[train]
    df_test = labels.iloc[test]
    X_train, X_val = train_test_split(
        df_train, test_size=0.1, stratify=df_train['Class'])
    X_train.to_csv(output_path + "train" +
                   str(i) + "Labels.csv", index=False)
    X_val.to_csv(output_path + "validation" +
                 str(i) + "Labels.csv", index=False)
    df_test.to_csv(output_path + "test" +
                   str(i) + "Labels.csv", index=False)
    i = i + 1
