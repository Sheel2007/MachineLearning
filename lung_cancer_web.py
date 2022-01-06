import keras.models
from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import os

model = load_model('cancer.h5')

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    gender = request.form.get('a')
    age = request.form.get('b')
    smoking = request.form.get('c')
    yellow_fingers = request.form.get('d')
    anxiety = request.form.get('e')
    peer_pressure = request.form.get('f')
    chronic_disease = request.form.get('g')
    fatigue = request.form.get('h')
    allergy = request.form.get('i')
    wheezing = request.form.get('j')
    alcohol_consumption = request.form.get('k')
    coughing = request.form.get('l')
    shortness_of_breath = request.form.get('m')
    swallowing_difficulty = request.form.get('n')
    chest_pain = request.form.get('o')
    # preprocess the data
    if gender.lower() == 'male':
        gender = 1
    else:
        gender = 0

    if smoking.lower() == 'no':
        smoking = 1
    else:
        smoking = 2

    if yellow_fingers.lower() == 'no':
        yellow_fingers = 1
    else:
        yellow_fingers = 2

    if anxiety.lower() == 'no':
        anxiety = 1
    else:
        anxiety = 2

    if peer_pressure.lower() == 'no':
        peer_pressure = 1
    else:
        peer_pressure = 2

    if chronic_disease.lower() == 'no':
        chronic_disease = 1
    else:
        chronic_disease = 2

    if fatigue.lower() == 'no':
        fatigue = 1
    else:
        fatigue = 2

    if allergy.lower() == 'no':
        allergy = 1
    else:
        allergy = 2

    if wheezing.lower() == 'no':
        wheezing = 1
    else:
        wheezing = 2

    if alcohol_consumption.lower() == 'no':
        alcohol_consumption = 1
    else:
        alcohol_consumption = 2

    if coughing.lower() == 'no':
        coughing = 1
    else:
        coughing = 2

    if shortness_of_breath.lower() == 'no':
        shortness_of_breath = 1
    else:
        shortness_of_breath = 2

    if swallowing_difficulty.lower() == 'no':
        swallowing_difficulty = 1
    else:
        swallowing_difficulty = 2

    if chest_pain.lower() == 'no':
        chest_pain = 1
    else:
        chest_pain = 2

    arr = np.array([[gender, age, smoking, yellow_fingers, anxiety, peer_pressure,
                     chronic_disease, fatigue, allergy, wheezing, alcohol_consumption,
                     coughing, shortness_of_breath, swallowing_difficulty, chest_pain]])

    int_arr = arr.astype(int)
  
    pred = model.predict(int_arr)

    if pred[0][0] > 1 - pred[0][1]:
        answer = 0
    else:
        answer = 1

    return render_template('after.html', data=answer)


if __name__ == '__main__':
    app.run(debug=True)
