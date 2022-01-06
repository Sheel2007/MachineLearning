import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten


df = pd.read_csv('Prostate_Cancer.csv')

df = df.drop('id', axis=1)
df["diagnosis_result"] = df["diagnosis_result"].map({"B": 0, "M": 1})#b is non cancerous

x = df.drop('diagnosis_result', axis=1)
y = df['diagnosis_result']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential([
    Flatten(input_shape=(x_train.shape[1], 1)),
    Dense(50, activation='relu'),
    Dense(50, activation='relu'),
    Dense(50, activation='relu'),
    Dense(25, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=27, epochs=50)

x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

model.evaluate(x_test, y_test)

model.save('prostate_cancer.h5')
