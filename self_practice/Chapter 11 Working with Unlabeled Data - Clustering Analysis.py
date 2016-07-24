# -*- coding: utf-8 -*-
# Chapter 11 Working with Unlabeled Data - Clustering Analysis

# Grouping objects by similarity using k-means
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=150,
                               n_features=2,
                               centers=3,
                               cluster_std=0.5,
                               shuffle=True,
                               random_state=0)

import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:,1], c='white', marker='o', s=50)
plt.grid()
plt.show()

# Apply k-means to our sample dataset using the KMeans class from scikit-learn's cluster module:
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3,
                        init='random',
                        n_init=10,
                        max_iter=300, 
                        tol=1e-04,
                        random_state=0)
y_km = km.fit_predict(X)

plt.scatter(X[y_km==0,0],
                X[y_km==0,1],
                s=50,
                c='lightgreen',
                marker='s',
                label='cluster 1')
plt.scatter(X[y_km==1,0],
                X[y_km==1,1],
                s=50,
                c='orange',
                marker='o',
                label='cluster 2')
plt.scatter(X[y_km==2,0],
                X[y_km==2,1],
                s=50,
                c='lightblue',
                marker='v',
                label='cluster 3')
plt.scatter(km.cluster_centers_[:,0],
                km.cluster_centers_[:,1],
                s=250,
                marker='*',
                c='red',
                label='centroids')
plt.legend()
plt.grid()
plt.show()
# Using the elbow method to find the optimal number of clusters
print 'Distortion: %.2f' % km.inertia_
distortions = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, 
                            init='k-means++',
                            n_init=10,
                            max_iter=300,
                            random_state=0)
    km.fit(X)
    distortions.append(km.inertia_)
plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()

# Quantifying the quality of clustering via silhouette plots
km = KMeans(n_clusters=3,
                        init='k-means++',
                        n_init=10,
                        max_iter=300,
                        tol=1e-04,
                        random_state=0)
y_km = km.fit_predict(X)

import numpy as np
from matplotlib import cm
from sklearn.metrics import silhouette_samples
cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X,
                                                            y_km,
                                                            metric='euclidean')
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(i / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper),
                 c_silhouette_vals,
                 height=1.0,
                 edgecolor='none',
                 color=color)
    yticks.append((y_ax_lower + y_ax_upper) / 2)
    y_ax_lower += len(c_silhouette_vals)
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg,
                 color="red",
                 linestyle="--")
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.show()

'''
To see how a silhouette plot looks for a relatively bad clustering, let's seed the
k-means algorithm with two centroids only:'''
km = KMeans(n_clusters=2,
                        init='k-means++',
                        n_init=10,
                        max_iter=300,
                        tol=1e-04,
                        random_state=0)
y_km = km.fit_predict(X)
plt.scatter(X[y_km==0,0],
                X[y_km==0,1],
                s=50, c='lightgreen',
                marker='s',
                label='cluster 1')
plt.scatter(X[y_km==1,0],
X[y_km==1,1],
s=50,
c='orange',
marker='o',
label='cluster 2')
plt.scatter(km.cluster_centers_[:,0],
                km.cluster_centers_[:,1],
                s=250,
                marker='*',
                c='red',
                label='centroids')
plt.legend()
plt.grid()
plt.show()

'''
Next we create the silhouette plot to evaluate the results. Please keep in mind that
we typically do not have the luxury of visualizing datasets in two-dimensional
scatterplots in real-world problems, since we typically work with data in higher
dimensions:
'''
cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X,
                                                            y_km,
                                                            metric='euclidean')
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(i / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper),
                 c_silhouette_vals,
                 height=1.0,
                 edgecolor='none',
                 color=color)
    yticks.append((y_ax_lower + y_ax_upper) / 2)
    y_ax_lower += len(c_silhouette_vals)
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color="red", linestyle="--")
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.show()

# Organizing clusters as a hierarchical tree
import pandas as pd
import numpy as np
np.random.seed(123)
variables = ['X', 'Y', 'Z']
labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']
X = np.random.random_sample([5, 3]) * 10
df = pd.DataFrame(X, columns=variables, index=labels)
df

# Performing hierarchical clustering on a distance matrix
from scipy.spatial.distance import pdist, squareform
row_dist = pd.DataFrame(squareform(pdist(df, metric='euclidean')), columns=labels, index=labels)
row_dist

from scipy.cluster.hierarchy import linkage
help(linkage)

#Incorrect approach
from scipy.cluster.hierarchy import linkage
row_clusters = linkage(row_dist, method='complete', metric='euclidean')
# Correct approach
row_clusters = linkage(pdist(df, metric='euclidean'), method='complete')
# Correct approach
row_clusters = linkage(df.values, method='complete', metric='euclidean')

pd.DataFrame(row_clusters, 
                         columns=['row label 1',
                                          'row label 2', 
                                          'distance',
                                          'no. of items in clust.'],
                          index=['cluster %d' %(i+1) for i in
                                      range(row_clusters.shape[0])])

from scipy.cluster.hierarchy import dendrogram
# make dendrogram black (part 1/2)
# from scipy.cluster.hierarchy import set_link_color_palette
# set_link_color_palette(['black'])
row_dendr = dendrogram(row_clusters, labels=labels, 
                                          # make dendrogram black (part 2/2)
                                          # color_threshold=npl.inf
                                          )           
plt.tight_layout()
plt.ylabel('Euclidean distance')
plt.show()

# Attaching dedrograms to a heat map
'''1. We create a new figure object and define the x axis position, y axis
position, width, and height of the dendrogram via the add_axes attribute.
Furthermore, we rotate the dendrogram 90 degrees counter-clockwise.
The code is as follows:'''
fig = plt.figure(figsize=(8,8), facecolor='white')
axd = fig.add_axes([0.09,0.1,0.2,0.6])
row_dendr = dendrogram(row_clusters, orientation='right')
# note: for matplotlib >= v1.5.1, please use orientation=‘left’
'''2. Next we reorder the data in our initial DataFrame according to the clustering
labels that can be accessed from the dendrogram object, which is essentially a
Python dictionary, via the leaves key. The code is as follows:'''
df_rowclust = df.ix[row_dendr['leaves'][::-1]]

'''3. Now we construct the heat map from the reordered DataFrame and position
it right next to the dendrogram:'''
axm = fig.add_axes([0.23,0.1,0.6,0.6])
cax = axm.matshow(df_rowclust,
                                  interpolation='nearest', cmap='hot_r')
'''4. Finally we will modify the aesthetics of the heat map by removing the axis
ticks and hiding the axis spines. Also, we will add a color bar and assign
the feature and sample names to the x and y axis tick labels, respectively.
The code is as follows:'''
axd.set_xticks([])
axd.set_yticks([])
for i in axd.spines.values():
    i.set_visible(False)
fig.colorbar(cax)
axm.set_xticklabels([''] + list(df_rowclust.columns))
axm.set_yticklabels([''] + list(df_rowclust.index))
plt.show()                     

# Applying agglomerative clustering via scikit-learn
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete')
labels = ac.fit_predict(X)
print 'Cluster labels: %s' % labels

# Locating regions of high density via DBSCAN
'''let's create a new dataset of half-moon-shaped structures to compare k-means clustering, 
hierarchical clustering, and DBSCAN:'''
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
plt.scatter(X[:, 0], X[:, 1])
plt.show()
'''We will start by using the k-means algorithm and complete linkage clustering to see
whether one of those previously discussed clustering algorithms can successfully
identify the half-moon shapes as separate clusters. The code is as follows:'''
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,3))
km = KMeans(n_clusters=2,
                        random_state=0)
y_km = km.fit_predict(X)
ax1.scatter(X[y_km==0,0],
                  X[y_km==0,1],
                  c='lightblue',
                  marker='o',
                  s=40,
                  label='cluster 1')
ax1.scatter(X[y_km==1,0],
                   X[y_km==1,1],
                   c='red',
                   marker='s',
                   s=40,
                   label='cluster 2')
ax1.set_title('K-means clustering')
ac = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete')
y_ac = ac.fit_predict(X)
ax2.scatter(X[y_ac==0, 0], 
                   X[y_ac==0, 1],
                   c='lightblue',
                   marker='o',
                   s=40,
                   label='cluster 1')
ax2.scatter(X[y_ac==0, 0], 
                   X[y_ac==0, 1],
                   c='red',
                   marker='s',
                   s=40,
                   label='cluster 2')
ax2.set_title('Agglomerative clustering')
plt.legend()
plt.show()

'''Finally, let's try the DBSCAN algorithm on this dataset to see if it can find the two
half-moon-shaped clusters using a density-based approach:'''
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.2, min_samples=5, metric='euclidean')
y_db = db.fit_predict(X)
plt.scatter(X[y_db==0, 0], X[y_db==0, 1], c='lightblue', marker='o', s=40, label='cluster 1')
plt.scatter(X[y_db==1, 0], X[y_db==1, 1], c='red', marker='s', s=40, label='cluster 2')
plt.legend()
plt.show()

