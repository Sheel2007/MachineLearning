import pandas as pd
import numpy as np

df = pd.read_csv('supermarket_sales - Sheet1.csv')

df = df.drop('Invoice ID', axis=1)
df = df.drop('Product line', axis=1)
df = df.drop('Date', axis=1)
df = df.drop('Time', axis=1)
