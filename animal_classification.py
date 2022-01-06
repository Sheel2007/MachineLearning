import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow import keras
import joblib


path = ('/content/datasets/')
data1 = pd.read_csv(path + 'zoo2.csv', index_col='animal_name')
data2 = pd.read_csv(path + 'zoo3.csv', index_col='animal_name')
df = pd.concat([data1, data2])

x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential([
    keras.layers.Flatten(input_shape=(x_train.shape[1], 1)),
    Dense(100, activation='relu'),
    Dense(50, activation='relu'),
    Dense(25, activation='relu'),
    Dense(10, activation='relu'),
    Dense(8, activation='softmax')
])

model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

model.fit(x_train, y_train, validation_split=0.1, batch_size=27, epochs=12)

x_test = np.array(x_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

model.evaluate(x_test, y_test)

joblib.dump(model, 'animals.joblib')


