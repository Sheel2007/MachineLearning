import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

df = pd.read_csv('segmentation data.csv')

df = df.drop('ID', axis=1)

print(df.columns)

values = ['Age', 'Income']

new_df = pd.DataFrame(df[values])

kmeans = KMeans(n_clusters=2)
kmeans.fit(new_df)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

color = ['red', 'green']

color = np.array(color)

color_num = [str(color[i]) for i in labels]

for i, _ in enumerate(color_num):
    plt.scatter(new_df[values[0]][i], new_df[values[1]][i], c=color_num[i])
plt.show()
