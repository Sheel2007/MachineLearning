import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow import keras
import joblib

df = pd.read_csv('datasets/healthcare-dataset-stroke-data.csv')

df = df.dropna() 

#split into info and labels
x = df.drop(["stroke"], axis=1)
y = df["stroke"]

#data preprocessing
x['smoking_status'] = x['smoking_status'].replace({'formerly smoked' or 'smokes':'smoked','never smoked' or 'Unknown':'non_smoking'})

x['smoking_status'] = [1 if i.strip() == 'smoked' else 0 for i in x.smoking_status]
x['gender'] = [1 if i.strip() == 'Male' else 0 for i in x.gender]
x['ever_married'] = [1 if i.strip() == 'Yes' else 0 for i in x.ever_married]
x['Residence_type'] = [1 if i.strip() == 'Urban' else 0 for i in x.Residence_type]

x = x.drop(["work_type"], axis=1)
x = x.drop(["id"], axis=1)

#splitting data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#convert to numpy arrays

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential([
    keras.layers.Flatten(input_shape=(x_train.shape[1], 1)),
    Dense(50, activation='relu'),
    Dense(25, activation='relu'),
    Dense(10, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

model.fit(x_train, y_train, validation_split=0.1, batch_size=10, epochs=20)

model.evaluate(x_test, y_test)

joblib.dump(model, 'stroke.joblib')
