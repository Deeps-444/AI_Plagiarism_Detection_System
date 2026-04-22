import joblib
from feature_extraction.features_script import extract_features 
from sklearn.preprocessing import StandardScaler

import numpy as np

model = joblib.load("../model/xgb_model.pkl")
scaler = joblib.load("../model/scaler.pkl")
def predict_fn(text):
    features = extract_features(text)
    features = scaler.transform([features])   # MUST scale
    prob = model.predict_proba(features)[0][1]
    return prob

# print(predict_fn("My day was great. I really hate fish and i didnt have to eat it."))
# print(predict_fn("Artificial intelligence is transforming modern society in unprecedented ways."))