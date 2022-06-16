
import numpy as np
import pandas as pd
import pickle
from flask import Flask, request

app=Flask(__name__)
pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)


@app.route('/')
def LogReg():
    return "Logistic Regression for Breast Cancer Wisconsin Data"

@app.route('/predict')
def predict_class():
    radius=request.args.get('radius')
    texture=request.args.get('texture')
    perimeter=request.args.get('perimeter')
    area=request.args.get('area')
    smoothness=request.args.get('smoothness')
    compactness=request.args.get('compactness')
    concavity=request.args.get('concavity')
    concave_points=request.args.get('concave_points')
    symmetry=request.args.get('symmetry')
    fractal_dimension=request.args.get('fractal_dimension')
    prediction=classifier.predict([[radius,texture,perimeter,area,smoothness,compactness,concavity,concave_points,symmetry,fractal_dimension]])
    return " The Predicated Class is"+ str(prediction)

@app.route('/predict_test', methods=["POST"])
def predict_test_class():
    df_test=pd.read_csv(request.files.get("file"))
    prediction=classifier.predict(df_test)
    return " The Prediction for the Breast Cancer Wisconsin dataset (Benign or Malignant) is"+ str(list(prediction))


if __name__=='__main__':
    app.run()