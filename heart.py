import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow import keras
import joblib

df = pd.read_csv('datasets/heart.csv')

hd = ['yes' if i == 1 else 'no' for i in df['target']]
df['Heart Disease'] = np.array(hd)

x = df.drop(['target', 'Heart Disease'], axis=1)
y = df['target']

x = pd.get_dummies(x, columns=['cp', 'restecg', 'slope', 'thal'])


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential([
    keras.layers.Flatten(input_shape=(x_train.shape[1], 1)),
    Dense(128, activation='relu'),
    Dense(50, activation='relu'),
    Dense(10, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])


model.fit(x_train, y_train, validation_split=0.1, batch_size=20, epochs=10)
model.evaluate(x_test, y_test)

joblib.dump(model, 'heart.joblib')