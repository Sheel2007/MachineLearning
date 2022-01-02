import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
import matplotlib.pyplot as plt

df = pd.read_csv('cardio_train.csv', sep=';')

df = df.drop('id', axis=1)

x = df.drop('cardio', axis=1)
y = df['cardio']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = Sequential([
    Flatten(input_shape=(x_train.shape[1], 1)),
    Dense(50, activation='relu'),
    Dense(50, activation='relu'),
    Dense(50, activation='relu'),
    Dense(4, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=20)

model.evaluate(x_test, y_test)