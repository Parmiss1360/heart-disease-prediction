import pickle as pk
import pandas as pd
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import utils as ut
# load processor
processor = ut.load_processor()
# load models
logistic_model = ut.load_model("logisticregression")
rf_model = ut.load_model("randomforest")
xgb_model = ut.load_model("xgboost")

#region variables for UI
sex=["female", "male"]
cp=["typical angina", "atypical angina", "non-anginal pain", "asymptomatic"]
restecg=["normal", "abnormal", "hypertrophy"]
exang=["no", "yes"]
slope=["upsloping", "flat", "downsloping"]
antal_ca = [0, 1, 2, 3]
fbs=["false", "true"]
thal=["normal", "fixed defect", "reversible defect"]
#endregion variables for UI


#region functions for UI
def validate_oldpeak(P):
    if P == "":
        return True
    try:
        float(P)   
        return True
    except:
        return False
def validate_age(new_value):
    if new_value == "":
        return True
    try:
        value = int(new_value)
        return 0 <= value <= 120
    except ValueError:
        return False

def validate(new_value):
    if new_value == "":
        return True
    if not new_value.isdigit():
        return False
    else:
        value = int(new_value)
        return True
def check_empty_fields():
    if age_entry.get() == "" or trestbps_entry.get() == "" or chol_entry.get() == "" or thalach_entry.get() == "" or oldpeak_entry.get() == "":
        messagebox.showerror("Error", "Please fill in all the fields.")
        return False
    return True
def check_parameters():
    
    if float(oldpeak_entry.get()) <0 or float(oldpeak_entry.get()) >6:
        resultlogistic_label.config(
            text="oldpeak must be between 0 and 6",
            fg="red"
        )
        return False
    if int(trestbps_entry.get()) <80 or int(trestbps_entry.get()) >200:
        resultlogistic_label.config(
            text="trestbps must be between 80 and 200",
            fg="red"
        )
        return False
    if int(chol_entry.get()) <100 or int(chol_entry.get()) >400:
        resultlogistic_label.config(
            text="chol must be between 100 and 400",
            fg="red"
        )
        return False
    if int(thalach_entry.get()) <70 or int(thalach_entry.get()) >210:
        resultlogistic_label.config(
            text="thalach must be between 70 and 210",
            fg="red"
        )
        return False
    return True

def get_processed_input():
    user_input = {
        "age": int(age_entry.get()),
        "sex": int(x.get()),
        "cp": int(cp_combobox.current()),
        "trestbps": int(trestbps_entry.get()),
        "chol": int(chol_entry.get()) if chol_entry.get() else 200,
        "fbs": int(fbs_var.get()),
        "restecg": int(restecg_combobox.current()),
        "thalach": int(thalach_entry.get()) if thalach_entry.get() else 150,
        "exang": int(exang_var.get()),
        "oldpeak": float(oldpeak_entry.get()) if oldpeak_entry.get() else 1.5,
        "slope": int(slope_combobox.current()),
        "ca": int(ca_combobox.current()),
        "thal": int(thal_combobox.current())
    }

    df = pd.DataFrame([user_input])
    df_processed = processor.transform(df)
    return df_processed   
def predict_logistic():
    if not check_empty_fields():
        return

    if not check_parameters():
        return
    df_processed = get_processed_input()
    pred_logistic = logistic_model.predict(df_processed)
    
    if pred_logistic[0] == 1:
        resultlogistic_label.config(
            text="logisticRegression: Has heart disease",
            fg="red"
        )
    else:
        resultlogistic_label.config(
            text="logisticRegression: No heart disease",
            fg="green"
        )

    
def predict_xgb():
    if not check_empty_fields():
        return

    if not check_parameters():
        return
 
    df_processed = get_processed_input()
    pred_xgb = xgb_model.predict(df_processed)
    if pred_xgb[0] == 1:
        resultxgb_label.config(
            text="XGBoost: Has heart disease",
            fg="red"
        )
    else:
        resultxgb_label.config(
            text="XGBoost: No heart disease",
            fg="green"
        )
    
def predict_random_forest():
    if not check_empty_fields():
        return

    if not check_parameters():
        return
    df_processed = get_processed_input()
    pred_random_forest = rf_model.predict(df_processed)
    if pred_random_forest[0] == 1:
        resultrf_label.config(
            text="Random Forest: Has heart disease",
            fg="red"
        )
    else:
        resultrf_label.config(
            text="Random Forest: No heart disease",
            fg="green"
        )
#endregion functions for UI

 
window= tk.Tk()
 
window.title("Heart Disease Prediction")

window.geometry('800x800')
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


# region sex 

sex_label = tk.Label(form, text="sex" , **label_style)
sex_label.grid(row=0, column=0)
sex_frame = tk.Frame(form)
sex_frame.grid(row=0, column=1, sticky="w")

x=tk.IntVar()
for index in range(len(sex)):
    sex_rb=tk.Radiobutton(sex_frame, text=sex[index], variable=x, value=index)
    sex_rb.pack(side="left", padx=5)
#endregion sex radiobuttons


#region fbs
fbs_label = tk.Label(form, text="fbs" , **label_style)
fbs_label.grid(row=1, column=0)
fbs_var = tk.IntVar()
fbs_frame = tk.Frame(form)
fbs_frame.grid(row=1, column=1, sticky="w")
for index in range(len(fbs)):
    fbs_rb=tk.Radiobutton(fbs_frame, text=fbs[index], variable=fbs_var, value=index)
    fbs_rb.pack(side="left", padx=5)
#endregion fbs radiobuttons

#region exang
exang_label = tk.Label(form, text="exang" , **label_style)
exang_label.grid(row=2, column=0)
exang_var = tk.IntVar()
exang_frame = tk.Frame(form)
exang_frame.grid(row=2, column=1, sticky="w")
for index in range(len(exang)):
    exang_rb=tk.Radiobutton(exang_frame, text=exang[index], variable=exang_var, value=index)
    exang_rb.pack(side="left", padx=5)
#endregion exang radiobuttons


#region UI age

age_entry = tk.Entry(master=form,width=20, font=("Arial", 10), bd=1, relief="solid", justify="center", 
                       validatecommand=(window.register(validate_age), '%P'), validate="key")

age_label = tk.Label(form,text="age" , **label_style)
age_label.grid(row=3, column=0, padx=10, pady=8, sticky="w")
age_entry.grid(row=3, column=1,padx=10, pady=8, sticky="w" )

#endregion UI age




#region UI trestbps

trestbps_entry = tk.Entry(master=form, font=("Arial", 10), bd=1, relief="solid", justify="center", 
                       validatecommand=(window.register(validate), '%P'), validate="key")

trestbps_label = tk.Label(form,text="trestbps" , **label_style)
trestbps_label.grid(row=4, column=0, padx=10, pady=8, sticky="w")
trestbps_entry.grid(row=4, column=1,padx=10, pady=8, sticky="w" )
tk.Label(form, text="(80–200)", fg="red").grid(row=4, column=2, sticky="w")
#endregion UI trestbps



#region UI chol

chol_entry = tk.Entry(master=form, font=("Arial", 10), bd=1, relief="solid", justify="center", 
                       validatecommand=(window.register(validate), '%P'), validate="key")

chol_label = tk.Label(form,text="chol" , **label_style)
chol_label.grid(row=5, column=0, padx=10, pady=8, sticky="w")
chol_entry.grid(row=5, column=1,padx=10, pady=8, sticky="w" )



tk.Label(form, text="(100–400)", fg="red").grid(row=5, column=2, sticky="w")
#endregion UI chol


#region UI oldpeak
oldpeak_entry = tk.Entry(master=form, font=("Arial", 10), bd=1, relief="solid", justify="center", 
                       validatecommand=(window.register(validate_oldpeak), '%P'), validate="key")

oldpeak_label = tk.Label(form,text="oldpeak" , **label_style)
oldpeak_label.grid(row=6, column=0, padx=10, pady=8, sticky="w")
oldpeak_entry.grid(row=6, column=1,padx=10, pady=8, sticky="w" )




tk.Label(form, text="(0–6)", fg="red").grid(row=6, column=2, sticky="w")
#endregion UI oldpeak


#region UI thalach

thalach_entry = tk.Entry(master=form, font=("Arial", 10), bd=1, relief="solid", justify="center", 
                       validatecommand=(window.register(validate), '%P'), validate="key")

thalach_label = tk.Label(form,text="thalach" , **label_style)
thalach_label.grid(row=7, column=0, padx=10, pady=8, sticky="w")
thalach_entry.grid(row=7, column=1,padx=10, pady=8, sticky="w" )





tk.Label(form, text="(70-210)", fg="red").grid(row=7, column=2, sticky="w")
#endregion UI thalach


#region UI cp

cp_default = tk.StringVar(value=cp[0])
cp_label = tk.Label(form, text="cp" , **label_style)
cp_label.grid(row=8, column=0)
cp_combobox=ttk.Combobox(form, textvariable=cp_default)
cp_combobox['values']=cp
cp_combobox.grid(row=8, column=1,padx=5, pady=8, sticky="w")
#endregion


#region UI ca
ca_default = tk.StringVar(value=antal_ca[0])
ca_label = tk.Label(form, text="ca" ,  **label_style)
ca_label.grid(row=9, column=0)
ca_combobox=ttk.Combobox(form, textvariable=ca_default)
ca_combobox['values']=antal_ca
ca_combobox.grid(row=9, column=1,padx=5, pady=8, sticky="w")
#endregion

#region UI thal

thal_default = tk.StringVar(value=thal[0])
thal_label = tk.Label(form, text="thal" ,  **label_style)
thal_label.grid(row=10, column=0)
thal_combobox=ttk.Combobox(form, textvariable=thal_default)
thal_combobox['values']=thal
thal_combobox.grid(row=10, column=1,padx=5, pady=8, sticky="w")
#endregion


#region UI restecg
restecg_default = tk.StringVar(value=restecg[0])
restecg_label = tk.Label(form, text="restecg" ,  **label_style)
restecg_label.grid(row=11, column=0)
restecg_combobox=ttk.Combobox(form, textvariable=restecg_default)
restecg_combobox['values']=restecg
restecg_combobox.grid(row=11, column=1,padx=5, pady=8, sticky="w")
#endregion

#region UI slope
slope_default = tk.StringVar(value=slope[0])
slope_label = tk.Label(form, text="slope" , **label_style)
slope_label.grid(row=12, column=0)
slope_combobox=ttk.Combobox(form, textvariable=slope_default)
slope_combobox['values']=slope
slope_combobox.grid(row=12, column=1,padx=5, pady=8, sticky="w")
#endregion

resultlogistic_label = tk.Label(form, text="", font=("Arial", 10, "bold"), fg="green")
resultlogistic_label.grid(row=13, column=3, columnspan=3, pady=15)

button=tk.Button(form, text="Predict_logistic", command=predict_logistic)
button.grid(row=13, column=0,columnspan=3, pady=15)


resultxgb_label = tk.Label(form, text="", font=("Arial", 10, "bold"), fg="green")
resultxgb_label.grid(row=14, column=3, columnspan=3, pady=15)

button=tk.Button(form, text="Predict_xgb", command=predict_xgb)
button.grid(row=14, column=0,columnspan=3, pady=15)


resultrf_label = tk.Label(form, text="", font=("Arial", 10, "bold"), fg="green")
resultrf_label.grid(row=15, column=3, columnspan=3, pady=15)

button=tk.Button(form, text="Predict_random_forest", command=predict_random_forest)
button.grid(row=15, column=0,columnspan=3, pady=15)

window.mainloop()

