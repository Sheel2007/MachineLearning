import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('heart_failure_clinical_records_dataset.csv')

colors = ['green', 'red']

color_num = [str(colors[i]) for i in df['DEATH_EVENT']]

for i, _ in enumerate(color_num):
    plt.scatter(df['age'][i], df['platelets'][i], c=color_num[i])

plt.show()
