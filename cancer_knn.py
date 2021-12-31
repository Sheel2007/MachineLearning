import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

df = pd.read_csv('KNNAlgorithmDataset.csv')

df = df.drop('Unnamed: 32', axis=1)

df['diagnosis'] = df['diagnosis'].replace({'M': 0, 'B': 1})

X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

X = X.drop('id', axis=1)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(x_train, y_train)

preds = knn.predict(x_test)

preds = np.array(preds)
y_test = np.array(y_test)

score = accuracy_score(y_test, preds)

print(df.columns)

colors = {0: 'green', 1: 'red'}

for i in range(len(df['texture_mean'])):
    plt.scatter(df['texture_mean'][i], df['smoothness_worst'][i], color=colors[df['diagnosis'][i]])

plt.show()
