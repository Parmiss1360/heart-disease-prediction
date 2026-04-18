import data_processing as dp
import model_training as mt 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv('./data/heart.csv')

dataprocessing = dp.Data_processing()
data = dataprocessing.delete_duplicates(data)
X = data.drop(columns=['target'])
Y = data['target']

x_train, x_test, y_train, y_test = dataprocessing.split_data(X, Y)
x_train = dataprocessing.fit_transform(x_train)
x_test  = dataprocessing.transform(x_test)


modelraining= mt.model_training()
modelraining.train_model_RandomForest(x_train, y_train)
modelraining.evalute(x_test, y_test)
print(metrics.f1_score(y_train, modelraining.predict(x_train)))
print(modelraining.feature_importance_Forestrandom(x_train))

