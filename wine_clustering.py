import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv('wine-clustering.csv')

df = df.dropna()

print(df.columns)

# print(df['Color_Intensity'].corr(df['Hue']))
'''
plt.scatter(df['Color_Intensity'], df['Hue'], c='red')
plt.show()
'''
kmeans = KMeans(n_clusters=3)

values = ['Flavanoids', 'Nonflavanoid_Phenols']

new_df = pd.DataFrame(df[values])

kmeans.fit(new_df)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

color = ['red', 'blue', 'green']

color = np.array(color)


color_num = [str(color[i]) for i in labels]
for i, _ in enumerate(color_num):
    plt.scatter(new_df[values[0]][i], new_df[values[1]][i], c=color_num[i])
plt.scatter(centroids[0][0], centroids[0][1], s=200, c='green')
plt.scatter(centroids[1][0], centroids[1][1], s=200, c='red')
plt.scatter(centroids[2][0], centroids[2][1], s=200, c='blue')
plt.show()
