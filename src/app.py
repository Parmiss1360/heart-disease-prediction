import pickle as pk
import pandas as pd

import tkinter as tk
from tkinter import ttk


# load processor

 
window= tk.Tk()
 
window.title("Heart Disease Prediction")

window.geometry('800x600')
window.mainloop()




user_input = {
    "age": 51,
    "sex": 0,
    "cp": 2,
    "trestbps": 140,
    "chol": 308,
    "fbs": 0,
    "restecg": 0,
    "thalach": 142,
    "exang": 0,
    "oldpeak": 1.5,
    "slope": 2,
    "ca": 1,
    "thal":2
}



with open("./models/processor.pkl", "rb") as f:
    processor = pk.load(f)
# load models
with open("./models/logisticregression_model.pkl", "rb") as f:
    logistic_model = pk.load(f)

with open("./models/randomforest_model.pkl", "rb") as f:
    rf_model = pk.load(f)

with open("./models/xgboost_model.pkl", "rb") as f:
    xgb_model = pk.load(f)
df = pd.DataFrame([user_input])

df_processed = processor.transform(df)

pred_logistic = logistic_model.predict(df_processed)
print("Logistic:", pred_logistic)

pred_xgb = xgb_model.predict(df_processed)
print("XGBoost:", pred_xgb)

pred_random_forest = rf_model.predict(df_processed)
print("Random Forest:", pred_random_forest)