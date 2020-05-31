from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
from tensorflow.keras import models
from sklearn.externals.joblib import dump, load

app = Flask(__name__)
sal_model = pickle.load(open('Poly_LR_Model.pkl', 'rb'))
churn_model = models.load_model('ChurnPredModel.h5')
churn_ss = load('std_scaler.bin')

@app.route('/')
def home():
    return render_template('MLHomePage.html')

@app.route('/SalaryPrediction')
def SalPred():
    return render_template('SalaryPrediction.html')

@app.route('/SalaryPrediction/Predict', methods=['POST'])
def predict():
    #using below - parameters will be stored in list
    int_features = [int(x) for x in request.form.values()]
    #below is to convert single parameter to integer. DO not use if multiple parameters
    res = sum(d * 10**i for i, d in enumerate(int_features[::-1]))
    input = np.array([1])
    featsqr = res*res
    featcube = res*res*res
    input = np.append(input,res)
    input = np.append(input,featsqr)
    input = np.append(input,featcube)
    #convert 1D array to 2D array.
    input = input.reshape(1,-1)
    prediction = sal_model.predict(input)
    output = round(prediction[0], 2)
    return render_template('SalaryPrediction.html', prediction_text='Salary at this level should be $ {}'.format(output))

@app.route('/ChurnPrediction')
def churnpred():
    return render_template('ChurnPrediction.html')

@app.route('/ChurnPrediction/Predict', methods=['POST'])
def churnpredict():
    #using below - parameters will be stored in list
    int_features = [int(x) for x in request.form.values()]
    int_features = np.array(int_features)
    #convert 1D array to 2D array.
    input = int_features.reshape(1,8)
    #do standard scaling of input
    input = churn_ss.transform(input)
    prediction = churn_model.predict(input)
    output = prediction>0.5
    return render_template('ChurnPrediction.html', prediction_text='Exited ? '.format(output), percent_text='Chances are : '.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)
