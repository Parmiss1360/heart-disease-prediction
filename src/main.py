import data_processing as dp
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
data=pd.read_csv('./data/heart.csv')

X=data.drop(columns=['target'])
Y=data['target']



dataprocessing = dp.Data_processing()
x_train, x_test, y_train, y_test = dataprocessing.split_data(X, Y)
x_train = dataprocessing.fit_transform(x_train)
x_test  = dataprocessing.transform(x_test)