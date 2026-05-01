import pandas as pd
import pickle as pk


def load_processor():
    with open("./models/processor.pkl", "rb") as f:
        processor = pk.load(f)
    return processor
def load_model(model_name):
    with open(f"./models/{model_name}_model.pkl", "rb") as f:
        model = pk.load(f)
    return model
def save_model(model, model_name):
        with open(f"./models/{model_name}_model.pkl", "wb") as f:
            pk.dump(model, f)
     
def save_processor(processor):
    with open("./models/processor.pkl", "wb") as f:
        pk.dump(processor, f)