import pickle as pk
import pandas as pd

import tkinter as tk
from tkinter import ttk

# load processor
with open("./models/processor.pkl", "rb") as f:
    processor = pk.load(f)
# load models
with open("./models/logisticregression_model.pkl", "rb") as f:
    logistic_model = pk.load(f)

with open("./models/randomforest_model.pkl", "rb") as f:
    rf_model = pk.load(f)

with open("./models/xgboost_model.pkl", "rb") as f:
    xgb_model = pk.load(f)
 
 
#region variables for UI
sex=["male", "female"]
cp=["typical angina", "atypical angina", "non-anginal pain", "asymptomatic"]
restecg=["normal", "abnormal", "hypertrophy"]
exang=["no", "yes"]
slope=["upsloping", "flat", "downsloping"]
antal_ca = [0, 1, 2, 3]
fbs=["false", "true"]
thal=["normal", "fixed defect", "reversible defect"]
#endregion variables for UI


#region functions for UI

def validate_age(new_value):
    if new_value == "":
        return True
    try:
        value = int(new_value)
        return 0 <= value <= 120
    except ValueError:
        return False

def validate_trestbps(new_value):
    if new_value == "":
        return True
    if not new_value.isdigit():
        return False
    try:
        value = int(new_value)
        return 0 <= value <= 200
    except ValueError:
        return False
    
def get_processed_input():
    user_input = {
        "age": int(age_entry.get()),
        "sex": int(x.get()),
        "cp": int(cp_combobox.current()),
        "trestbps": int(trestbps_entry.get()),
        "chol": 308,
        "fbs": int(fbs_var.get()),
        "restecg": int(restecg_combobox.current()),
        "thalach": 142,
        "exang": int(exang_var.get()),
        "oldpeak": 1.5,
        "slope": int(slope_combobox.current()),
        "ca": int(ca_combobox.current()),
        "thal": int(thal_combobox.current())
    }

    df = pd.DataFrame([user_input])
    df_processed = processor.transform(df)
    return df_processed   
def predict_logistic():
    df_processed = get_processed_input()
    pred_logistic = logistic_model.predict(df_processed)

    if pred_logistic[0] == 1:
        resultlogistic_label.config(
            text="logisticRegression: Has heart disease",
            fg="red"
        )
    else:
        resultlogistic_label.config(
            text="No heart disease",
            fg="green"
        )

    
def predict_xgb():
    df_processed = get_processed_input()
    pred_xgb = xgb_model.predict(df_processed)
    text=f"XGBoost: {'skicked' if pred_xgb[0] == 1 else 'not skicked'}"
    #resultxgb_label.config(text=text)
    
def predict_random_forest():
    df_processed = get_processed_input()
    pred_random_forest = rf_model.predict(df_processed)
    text=f"Random Forest: {'skicked' if pred_random_forest[0] == 1 else 'not skicked'}"
    #resultrandomforest_label.config(text=text)
#endregion functions for UI

 
window= tk.Tk()
 
window.title("Heart Disease Prediction")

window.geometry('600x600')
form = tk.Frame(window, padx=10, pady=10)
form.grid(row=0, column=0, sticky="nw")

form.columnconfigure(0, weight=0)
form.columnconfigure(1, weight=0)
form.columnconfigure(2, weight=0)

label_style = {
    "fg": "blue",
    "font": ("Arial", 10),
    "width": 7,
    "anchor": "w"
}
#region UI age

age_entry = tk.Entry(master=form,width=20, font=("Arial", 10), bd=1, relief="solid", justify="center", 
                       validatecommand=(window.register(validate_age), '%P'), validate="key")

age_label = tk.Label(form,text="age" , **label_style)
age_label.grid(row=0, column=0, padx=10, pady=8, sticky="w")
age_entry.grid(row=0, column=1,padx=10, pady=8, sticky="w" )

#endregion UI age


resultlogistic_label = tk.Label(form, text="", font=("Arial", 12, "bold"), fg="green")
resultlogistic_label.grid(row=10, column=4, columnspan=3, pady=15)

button=tk.Button(form, text="Predict_logistic", command=predict_logistic)
button.grid(row=10, column=0,columnspan=3, pady=15)



# region sex 

sex_label = tk.Label(form, text="sex" , **label_style)
sex_label.grid(row=1, column=0)
sex_frame = tk.Frame(form)
sex_frame.grid(row=1, column=1, sticky="w")

x=tk.IntVar()
for index in range(len(sex)):
    sex_rb=tk.Radiobutton(sex_frame, text=sex[index], variable=x, value=index)
    sex_rb.pack(side="left", padx=5)
#endregion sex radiobuttons


#region fbs
fbs_label = tk.Label(form, text="fbs" , **label_style)
fbs_label.grid(row=2, column=0)
fbs_var = tk.IntVar()
fbs_frame = tk.Frame(form)
fbs_frame.grid(row=2, column=1, sticky="w")
for index in range(len(fbs)):
    fbs_rb=tk.Radiobutton(fbs_frame, text=fbs[index], variable=fbs_var, value=index)
    fbs_rb.pack(side="left", padx=5)
#endregion fbs radiobuttons

#region exang
exang_label = tk.Label(form, text="exang" , **label_style)
exang_label.grid(row=3, column=0)
exang_var = tk.IntVar()
exang_frame = tk.Frame(form)
exang_frame.grid(row=3, column=1, sticky="w")
for index in range(len(exang)):
    exang_rb=tk.Radiobutton(exang_frame, text=exang[index], variable=exang_var, value=index)
    exang_rb.pack(side="left", padx=5)
#endregion exang radiobuttons

#region UI cp

cp_default = tk.StringVar(value=cp[0])
cp_label = tk.Label(form, text="cp" , **label_style)
cp_label.grid(row=4, column=0)
cp_combobox=ttk.Combobox(form, textvariable=cp_default)
cp_combobox['values']=cp
cp_combobox.grid(row=4, column=1,padx=5, pady=8, sticky="w")
#endregion


#region UI ca
ca_default = tk.StringVar(value=antal_ca[0])
ca_label = tk.Label(form, text="ca" ,  **label_style)
ca_label.grid(row=5, column=0)
ca_combobox=ttk.Combobox(form, textvariable=ca_default)
ca_combobox['values']=antal_ca
ca_combobox.grid(row=5, column=1,padx=5, pady=8, sticky="w")
#endregion

#region UI thal

thal_default = tk.StringVar(value=thal[0])
thal_label = tk.Label(form, text="thal" ,  **label_style)
thal_label.grid(row=6, column=0)
thal_combobox=ttk.Combobox(form, textvariable=thal_default)
thal_combobox['values']=thal
thal_combobox.grid(row=6, column=1,padx=5, pady=8, sticky="w")
#endregion


#region UI restecg
restecg_default = tk.StringVar(value=restecg[0])
restecg_label = tk.Label(form, text="restecg" ,  **label_style)
restecg_label.grid(row=7, column=0)
restecg_combobox=ttk.Combobox(form, textvariable=restecg_default)
restecg_combobox['values']=restecg
restecg_combobox.grid(row=7, column=1,padx=5, pady=8, sticky="w")
#endregion

#region UI slope
slope_default = tk.StringVar(value=slope[0])
slope_label = tk.Label(form, text="slope" , **label_style)
slope_label.grid(row=8, column=0)
slope_combobox=ttk.Combobox(form, textvariable=slope_default)
slope_combobox['values']=slope
slope_combobox.grid(row=8, column=1,padx=5, pady=8, sticky="w")
#endregion

#region UI trestbps

trestbps_entry = tk.Entry(master=form, font=("Arial", 10), bd=1, relief="solid", justify="center", 
                       validatecommand=(window.register(validate_trestbps), '%P'), validate="key")

trestbps_label = tk.Label(form,text="trestbps" , **label_style)
trestbps_label.grid(row=9, column=0, padx=10, pady=8, sticky="w")
trestbps_entry.grid(row=9, column=1,padx=10, pady=8, sticky="w" )

#endregion UI trestbps
window.mainloop()

