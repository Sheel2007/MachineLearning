import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

df = pd.read_table('fruit_data_with_colors.txt')

# print(df['fruit_subtype'].value_counts())

df = df.drop('fruit_name', axis=1)
df = df.drop('fruit_subtype', axis=1)

print(df.head())

X = df.drop('fruit_label', axis=1)
y = df['fruit_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

preds = knn.predict(X_test)

print(preds)

preds = np.array(preds)
y_test = np.array(y_test)

print(y_test)

score = accuracy_score(y_test, preds)
print(score)

color = ['green', 'red']
