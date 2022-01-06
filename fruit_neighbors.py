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

print(df.columns)

X = df.drop('fruit_label', axis=1)
y = df['fruit_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

preds = knn.predict(X_test)

preds = np.array(preds)
y_test = np.array(y_test)

score = accuracy_score(y_test, preds)

color = ['green', 'red']
#plotting accuracy
for i, _ in enumerate(y_test):
    if preds[i] == y_test[i]:
        plt.scatter(y_test[i], preds[i], c=color[0])
    else:
        plt.scatter(y_test[i], preds[i], c=color[1])
plt.show()
#plotting other values
for i, _ in enumerate(y_test):
    if preds[i] == y_test[i]:
        plt.scatter(df['width'][i], df['height'][i], c=color[0])
    else:
        plt.scatter(df['height'][i], df['width'][i], c=color[1])

plt.show()
