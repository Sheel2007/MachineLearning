import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow import keras

df = pd.read_csv('parkinsons2.csv')

x = df.drop('status', axis=1)
y = df['status']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential([
    keras.layers.Flatten(input_shape=(x_train.shape[1], 1)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=14) 

model.evaluate(x_test, y_test) #I tested and got 84%
