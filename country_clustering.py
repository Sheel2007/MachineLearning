import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

df = pd.read_csv('Country-data.csv')

df = df.drop('country', axis=1)

# filtered_label0 = df[labels == 0]

# for i in cluster:
# print(cluster[i], labels[i])
# plt.scatter(cluster[i][0], labels[i])

# print(df.dropna)

# print(df.describe())
# print(df.shape)
# print(df.columns)


# print(new_df)


"""
for i, index in enumerate(labels):
    if index == 0:
        labels[index] = str(labels[index])
        labels[index] = 'blue'
    elif index == 1:
        labels[index] = str(labels[index])
        labels[index] = 'red'
"""
# print(type(color_num[2]))

"""
for i, _ in enumerate(color_num):
    plt.scatter(new_df['child_mort'][i], new_df['exports'][i], c=color_num[i][2])
plt.scatter(cluster[0][0], cluster[0][1], s=200, c='blue')
plt.scatter(cluster[1][0], cluster[1][1], s=200, c='red')
plt.show()
"""


# print(df['child_mort'])


def scatter_plot(col1, col2, color, labels, cluster):
    color = np.array(color)

    color_num = [str(color[i]) for i in labels]
    for i, _ in enumerate(color_num):
        plt.scatter(col1[i], col2[i], c=color_num[i])
    plt.scatter(cluster[0][0], cluster[0][1], s=200, c='blue')
    plt.scatter(cluster[1][0], cluster[1][1], s=200, c='red')
    plt.show()


values = ['imports', 'gdpp']

new_df = pd.DataFrame(df[values])

new_df = new_df.dropna()
kmeans = KMeans(n_clusters=2)
kmeans.fit(new_df)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

color = ['red', 'blue']

scatter_plot(new_df[values[0]], new_df[values[1]], color, labels, centroids)
