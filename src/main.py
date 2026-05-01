import data_processing as dp
import model_training as mt 
import pandas as pd 
import utils as ut

data = pd.read_csv('./data/heart.csv')

dataprocessing = dp.Data_processing()

#Denna metod tar bor felakitg värde och ersätta värde med mode.
data = dataprocessing.replace_missing_values(data)

#för att kunna använda dataprocessing i UI:n så sparar jag den som en pickle fil

data = dataprocessing.delete_duplicates(data)


X = data.drop(columns=['target'])
Y = data['target']

x_train, x_test, y_train, y_test = dataprocessing.split_data(X, Y)
x_train = dataprocessing.fit_transform(x_train)
x_test  = dataprocessing.transform(x_test)
model_tarining = mt.model_training()


ut.save_processor(dataprocessing)
    
results = []

model_tarining.train_model_XGBoost(x_train, y_train)
r = model_tarining.evalute(x_test, y_test)
#för att kunna använda modellen i UI:n så sparar jag den som en pickle fil
ut.save_model(model_tarining.model, "xgboost")
r["model"] = "XGBoost"
results.append(r)

model_tarining.train_model_LogisticRegression(x_train, y_train)
r = model_tarining.evalute(x_test, y_test)
#för att kunna använda modellen i UI:n så sparar jag den som en pickle fil
ut.save_model(model_tarining.model, "logisticregression")
r["model"] = "LogisticRegression"
results.append(r)

model_tarining.train_model_RandomForest(x_train, y_train)
r = model_tarining.evalute(x_test, y_test)
#för att kunna använda modellen i UI:n så sparar jag den som en pickle fil
ut.save_model(model_tarining.model, "randomforest")

r["model"] = "RandomForest"
results.append(r)

print(pd.DataFrame(results))

    