import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv('CC GENERAL.csv')

df = df.drop('CUST_ID', axis=1)

df = df.dropna()

print(df.columns)

values = ['BALANCE', 'CREDIT_LIMIT']

new_df = pd.DataFrame(df[values])

kmeans = KMeans(n_clusters=2)
kmeans.fit(new_df)

labels = kmeans.labels_

centroids = kmeans.cluster_centers_

plt.scatter(df[values[0]], df[values[1]])
plt.scatter(centroids[0][0], centroids[0][1], s=200)
plt.scatter(centroids[1][0], centroids[1][1], s=200)
plt.show()
