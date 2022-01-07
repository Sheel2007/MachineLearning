import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv('Airplane_Crashes_and_Fatalities_Since_1908.csv')

df = pd.DataFrame(df[['Aboard', 'Fatalities', 'Ground']])

df = df.dropna()

kmeans = KMeans(n_clusters=2)
kmeans.fit(df)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

colors = ['green', 'red']
values = ['Aboard', 'Ground']

color_num = [str(colors[i]) for i in labels]

for i, _ in enumerate(color_num):
    try:
        plt.scatter(df[values[0]][i], df[values[1]][i], c=color_num[i])
    except:
        print(f'Row: {i} contains NaN')
plt.scatter(centroids[0][0], centroids[0][1], s=200, c='blue')
plt.scatter(centroids[1][0], centroids[1][1], s=200, c='red')
plt.show()
