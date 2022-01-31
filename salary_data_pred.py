import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('Salary_data.csv')

x = df.drop('Salary', axis=1)
y = df['Salary']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

lr = LinearRegression()

lr.fit(x_train, y_train)

score = lr.score(x_test, y_test)
print(score)

predictions = lr.predict(x)

x = np.array(x)
y = np.array(y)
predictions = np.array(predictions)

plt.scatter(x, y)
plt.plot(x, predictions, color='green')
plt.show()