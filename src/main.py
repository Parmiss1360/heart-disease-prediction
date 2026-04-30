import data_processing as dp
import model_training as mt 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

import pickle as pk

import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv('./data/heart.csv')

dataprocessing = dp.Data_processing()

#för att kunna använda dataprocessing i UI:n så sparar jag den som en pickle fil

data = dataprocessing.delete_duplicates(data)

X = data.drop(columns=['target'])
Y = data['target']

x_train, x_test, y_train, y_test = dataprocessing.split_data(X, Y)
x_train = dataprocessing.fit_transform(x_train)
x_test  = dataprocessing.transform(x_test)
model_tarining = mt.model_training()

with open("./models/processor.pkl", "wb") as f:
    pk.dump(dataprocessing, f)
    
results = []

model_tarining.train_model_XGBoost(x_train, y_train)
r = model_tarining.evalute(x_test, y_test)
#för att kunna använda modellen i UI:n så sparar jag den som en pickle fil
with open("./models/xgboost_model.pkl", "wb") as f:
    pk.dump(model_tarining.model, f)
r["model"] = "XGBoost"
results.append(r)

model_tarining.train_model_LogisticRegression(x_train, y_train)
r = model_tarining.evalute(x_test, y_test)
#för att kunna använda modellen i UI:n så sparar jag den som en pickle fil
with open("./models/logisticregression_model.pkl", "wb") as f:
    pk.dump(model_tarining.model, f)
r["model"] = "LogisticRegression"
results.append(r)

model_tarining.train_model_RandomForest(x_train, y_train)
r = model_tarining.evalute(x_test, y_test)
#för att kunna använda modellen i UI:n så sparar jag den som en pickle fil
with open("./models/randomforest_model.pkl", "wb") as f:
    pk.dump(model_tarining.model, f)
r["model"] = "RandomForest"
results.append(r)

print(pd.DataFrame(results))

    