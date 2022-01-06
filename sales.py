import pandas as pd
import numpy as np

df = pd.read_csv('supermarket_sales - Sheet1.csv')

df = df.drop('Invoice ID', axis=1)
df = df.drop('Product line', axis=1)
df = df.drop('Date', axis=1)
df = df.drop('Time', axis=1)

df['Branch'] = df['Branch'].replace({'A': 0, 'B': 1, 'C': 2})
df['City'] = df['City'].replace({'Yangon': 0, 'Naypyitaw': 1, 'Mandalay': 2})
df['Customer type'] = df['Customer type'].replace({'Normal': 0, 'Member': 1})
df['Gender'] = df['Gender'].replace({'Female': 0, 'Male': 1})
df['Payment'] = df['Payment'].replace({'Cash': 0, 'Credit card': 1, 'Ewallet': 2})
