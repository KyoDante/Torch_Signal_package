from sklearn.cluster import k_means


# n_clusters : 多少个类
def make_kmeans(data, n_clusters=3):
    # 中心 centroid，类似data的数据排布
    # 类别 labels，分簇之后的标类
    centroid, labels, _, _ = k_means(data, n_clusters)
    return centroid, labels
