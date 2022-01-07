import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

df = pd.read_csv('heart_failure_clinical_records_dataset.csv')

colors = ['green', 'red']

color_num = [str(colors[i]) for i in df['DEATH_EVENT']]

for i, _ in enumerate(color_num):
    plt.scatter(df['age'][i], df['platelets'][i], c=color_num[i])

plt.show()

x = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

clf = SVC()
clf.fit(x_train, y_train)
