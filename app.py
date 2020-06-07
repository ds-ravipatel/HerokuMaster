from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
from tensorflow.keras import models
from sklearn.externals.joblib import dump, load
import tensorflow as tf
from tensorflow import keras
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

app = Flask(__name__)


def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    keras.backend.get_session().run(tf.local_variables_initializer())
    return auc

sal_model = pickle.load(open('Poly_LR_Model.pkl', 'rb'))
# load the model, and pass in the custom metric function
#global graph
graph = tf.get_default_graph()
churn_model = models.load_model('new_ChurnPredModel.h5', custom_objects={'auc': auc})

#churn_model = models.load_model('ChurnPredModel.h5')
churn_ss = load('new_std_scaler.bin')

#below is loaded for NLP- restaurant reviews
dt_loaded = load('dt_model.pkl')
cv_loaded = load('cv_obj.pkl')
stop_words = stopwords.words('english')
#

@app.route('/')
def home():
    return render_template('index.html')

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

@app.route('/RestaurantReviewPrediction')
def RReviewPred():
    return render_template('RestaurantReviewPrediction.html')

@app.route('/RestaurantReviewPrediction/Predict', methods=['POST'])
def RReviewPrediction():
    #using below - parameters will be stored in variable
    r = ""
    r = [str(x) for x in request.form.values()]
    r =  r[0]
    l = []
    r = r.lower()
    r = re.sub('[^a-z0-9]+', ' ', r)
    res_words = []
    ps = PorterStemmer()
    res_words = [ps.stem(word) for word in r.split() if word not in stop_words]
    r = ' '.join(res_words)
    l = [r]
    prediction =-1
    prediction = dt_loaded.predict(cv_loaded.transform(l).toarray())[0]
    return render_template('RestaurantReviewPrediction.html', prediction = prediction)

@app.route('/ChurnPrediction')
def churnpred():
    return render_template('ChurnPrediction.html')

@app.route('/ChurnPrediction/Predict', methods=['POST'])
def churnpredict():
    #using below - parameters will be stored in list
    int_features = [int(x) for x in request.form.values()]
    int_features = np.array(int_features)
    #convert 1D array to 2D array.
    input = int_features.reshape(1,-1)
    #do standard scaling of input
    input = churn_ss.transform(input)
    with graph.as_default():
        prediction = churn_model.predict(input)
        prediction = prediction[0][0]
        output = prediction>0.5
    return render_template('ChurnPrediction.html', prediction_text='Exited ? '.format(output), percent_text='Chances are : '.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)