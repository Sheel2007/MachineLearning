import keras.models
from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import os

model = load_model('prostate_cancer.h5')

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('prostate_cancer_home.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    radius = request.form.get('a')
    texture = request.form.get('b')
    perimeter = request.form.get('c')
    area = request.form.get('d')
    smoothness = request.form.get('e')
    compactness = request.form.get('f')
    symmetry = request.form.get('g')
    fractal_dimension = request.form.get('h')
    # preprocess the data
    arr = np.array([[radius, texture, perimeter, area,
                     smoothness, compactness, symmetry, fractal_dimension]])

    int_arr = arr.astype(float)

    pred = model.predict(int_arr)

    print(pred)

    if pred[0][0] > 1 - pred[0][1]:
        answer = 0
    else:
        answer = 1

    return render_template('prostate_cancer_after.html', data=answer)


if __name__ == '__main__':
    app.run(debug=True)
