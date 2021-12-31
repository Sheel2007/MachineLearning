import keras.models
from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import os

model = load_model('breast_cancer.h5')

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('breast_cancer_home.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    mean_radius = request.form.get('a')
    mean_texture = request.form.get('b')
    mean_perimeter = request.form.get('c')
    mean_area = request.form.get('d')
    mean_smoothness = request.form.get('o')
    # preprocess the data

    arr = np.array([[mean_radius, mean_texture, mean_perimeter,
                     mean_area, mean_smoothness]])

    int_arr = arr.astype(float)

    pred = model.predict(int_arr)

    print(pred)

    if pred[0][0] > 1 - pred[0][1]:
        answer = 0
    else:
        answer = 1

    return render_template('breast_cancer_after.html', data=answer)


if __name__ == '__main__':
    app.run(debug=True)
