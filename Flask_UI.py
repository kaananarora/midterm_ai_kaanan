

import numpy as np
import pandas as pd
import pickle
import pandas as pd
from flask import Flask, request
import flasgger
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)

pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)


@app.route('/')
def LogReg():
    return "Logistic Regression for Breast Cancer Wisconsin Data"


@app.route('/predict_test', methods=["POST"])
def predict_test_class():
    
    """predicting if  for breast cancer is benign of malignant
    ---
    parameters:  
      - name: file
        in: formData
        type: file
        required: true
    responses:
        500:
            description: The output values
        
    """
    df_test=pd.read_csv(request.files.get("file"))
    prediction=classifier.predict(df_test)
    return "The Prediction for the Breast Cancer Wisconsin dataset (Benign or Malignant) is"+ str(list(prediction))



if __name__=='__main__':
    app.run()