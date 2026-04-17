import pandas as pd
import joblib

def load_data(path):
    return pd.read_csv(path)

def save_model(model, path):
    joblib.dump(model, path)

def load_model_file(path):
    return joblib.load(path)