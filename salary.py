import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('Levels_Fyi_Salary_Data.csv')

df = df[['company', 'totalyearlycompensation', 'yearsofexperience', 'yearsatcompany', 'basesalary']]

x = df.drop('totalyearlycompensation', axis=1)
y = df['totalyearlycompensation']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

lr = LinearRegression()

lr.fit(x_train, y_train)

