import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow import keras
from sklearn.model_selection import train_test_split

df = pd.read_csv('survey lung cancer.csv')
#1 means no 2 means yes

df["GENDER"] = df["GENDER"].map({"F": 0, "M": 1})
df['LUNG_CANCER']= df['LUNG_CANCER'].map({'NO':0, "YES":1})

x = df.drop('LUNG_CANCER', axis=1)
y = df['LUNG_CANCER']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential([
    keras.layers.Flatten(input_shape=(x_train.shape[1], 1)),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
            
model.fit(x_train, y_train, validation_split=0.1, batch_size=10, epochs=10)

model.evaluate(x_test, y_test)

model.save('cancer.h5')
