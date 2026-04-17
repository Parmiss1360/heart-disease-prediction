
# import sklearn.datasets as datasets
# import pandas as pd
# import pickle  as pkl
# iris=datasets.load_iris()


# df=pd.DataFrame(iris.data,columns=iris.feature_names)
# y=iris.target
# from sklearn.tree import DecisionTreeClassifier
# dtree=DecisionTreeClassifier()
# dtree.fit(df,y)
# with open('model.pkl','wb') as f:
#     pkl.dump(dtree,f)
    
# with open('model.pkl','rb') as f:
#     dtree=pkl.load(f)


# preds=dtree.predict([[5.1, 3.5, 1.4, 0.2],[6.2, 3.4, 5.4, 2.3]])


import sklearn.datasets as datasets
import pandas as pd
import pickle  as pkl
from sklearn.model_selection import train_test_split




df = pd.read_csv("./data/heart.csv") 
y = df["target"]
x = df.iloc[:, :-1]
print(x)

print(pd.Series(y).value_counts()) 

# iris=datasets.load_iris()

# df=pd.DataFrame(iris.data,columns=iris.feature_names)
# y=iris.target



x_train,x_test,y_train,y_test=train_test_split(df,y,test_size=0.2,random_state=42, stratify=y)
print("Train:")
print(y_train.value_counts(normalize=True))

print("Test:")
print(y_test.value_counts(normalize=True))    
# from sklearn.preprocessing import StandardScaler

# SS=StandardScaler()

# df_scaled=SS.fit_transform(x_train)



# from sklearn.tree import DecisionTreeClassifier
# dtree=DecisionTreeClassifier()
# dtree.fit(df_scaled,y_train)
# with open('model.pkl','wb') as f:
#     pkl.dump(dtree,f)
    
# with open('model.pkl','rb') as f:
#     dtree2=pkl.load(f)




# from sklearn import metrics
# x_test_scaled=SS.transform(x_test)
# pred=dtree.predict(x_test_scaled)
# print(metrics.classification_report(y_test,pred))

# #preds=dtree2.predict([[5.1, 3.5, 1.4, 0.2],[6.2, 3.4, 5.4, 2.3]])
# print(y_test)
# print(pred)
# # print(iris.target_names[preds])