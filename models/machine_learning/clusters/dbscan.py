from sklearn.cluster import DBSCAN

# ref: https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py
def make_dbscan(data):
    db = DBSCAN(eps=0.3, min_samples=5)
    db = db.fit(data)

    labels = db.labels

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print(f'Estimated number of clusters: {n_clusters_}')

    return labels