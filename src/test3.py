import pandas  as pd
import numpy   as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import pickle   as pk
import seaborn   as sns 

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
#data=pd.read_csv('./data/diabetes.csv')




# x=data.iloc[:,3:4].values
# si=SimpleImputer(missing_values=0, strategy='mean')
# x=si.fit_transform(x)
# data['skin']=x
# print(data [])

# plt.hist(data['test'], bins=20)
# plt.xlabel('Test    Values')      
# plt.show()
# skin_mean=data[data['skin']!=0].mean()['skin']
# data.replace( {'skin':0}, inplace=True, value=skin_mean)

# test_mean=data[data['test']!=0].mean()['test']
# data.replace( {'test':0}, inplace=True, value=test_mean)

# data.replace( {'skin':0,'test':0 }, inplace=True, value=np.nan)
# x=data.iloc[:,:-1].values
# test_imputer = KNNImputer(n_neighbors=3)
# data=test_imputer.fit_transform(data)
# # with open('test_imputer.pkl', 'wb') as f:
# #     pk.dump(test_imputer, f)
# # with open('test_imputer.pkl', 'rb') as f:
# #     test_imputer = pk.load(f)

# # # print(test_imputer.transform([[0.0  ,137 , 40,  np.nan,  168.000000,  43.1 , 2.288,  33  ]]))
# data = data[:, :-1]
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler(feature_range=(0, 1))
# data = scaler.fit_transform(data)

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# data = scaler.fit_transform(data)

# from sklearn.preprocessing import Normalizer
# norm = Normalizer()
# data = norm.fit_transform(data)

# print(np.sum(data[0]**2))

#////
import scipy.stats as stats

data=pd.read_csv('./data/heart.csv')

#data['chol_sqrt'] = np.log(data['chol'])


# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# data['chol_scaled'] = scaler.fit_transform(data[['chol']])  
# print(data[(data['chol_scaled'] < -3) | (data['chol_scaled'] > 3)]['chol_scaled'])

# #data2=data[(data['thalach_scaled'] >= -3) & (data['thalach_scaled'] <= 3)].reset_index()

# #print(data2)

# q1 = data['chol'].quantile(0.25)
# q3 = data['chol'].quantile(0.75)
# IQR=q3-q1
# lowerbound=q1-(1.5*IQR)
# upperbound=q3+(1.5*IQR)
# print("Lower bound:", lowerbound)   
# print("Upper bound:", upperbound)

# #print(data[(data['chol'] < lowerbound) | (data['chol'] > upperbound)])

# data=data[(data['chol'] >= lowerbound) & (data['chol'] <= upperbound)].reset_index(drop=True)
# #print(data)
# q1 = data['chol'].quantile(0.25)
# q3 = data['chol'].quantile(0.75)
# IQR=q3-q1
# lowerbound=q1-(1.5*IQR)
# upperbound=q3+(1.5*IQR)
# print("Lower bound:", lowerbound)   
# print("Upper bound:", upperbound)

# print(data[(data['chol'] < lowerbound) | (data['chol'] > upperbound)])

import data_processing as dp
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
data=pd.read_csv('./data/heart.csv')




dataprocessing = dp.Data_processing()
data = dataprocessing.delete_duplicates(data)
X=data.drop(columns=['target'])
Y=data['target']
x_train, x_test, y_train, y_test = dataprocessing.split_data(X, Y)
x_train = dataprocessing.fit_transform(x_train)
x_test  = dataprocessing.transform(x_test)

# print("Train:Balancded")
# print(y_train.value_counts(normalize=True))

# print("Test:Balanced")
# print(y_test.value_counts(normalize=True))    

# logreg=LogisticRegression()
# logreg.fit(x_train,y_train)
# pred=logreg.predict(x_test)
# print(metrics.classification_report(y_test,pred))
# print(metrics.confusion_matrix(y_test,pred))


rfc = RandomForestClassifier(
    n_estimators=30,
    max_depth=5,
    random_state=42
)
rfc.fit(x_train,y_train)
pred=rfc.predict(x_test)
pred_train=rfc.predict(x_train)
print(metrics.accuracy_score(y_test, pred))
print(metrics.accuracy_score(y_train, pred_train))

# print(metrics.classification_report(y_test,pred))
# print(metrics.confusion_matrix(y_test,pred))