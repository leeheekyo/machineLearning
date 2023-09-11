from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

x = np.array([[4, 2], [1, 4], [1, 0],
    [9, 2], [6, 4], [6, 0]])
kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(x)
print(kmeans.labels_)
kmeans.predict([[0, 0], [12, 3]])
print(kmeans.cluster_centers_)

plt.scatter(x[:,0], x[:,1], c=kmeans.labels_, marker = 'o')
plt.scatter(x[:,0], x[:,1], c=kmeans.labels_, marker = 'x')

plt.show()
