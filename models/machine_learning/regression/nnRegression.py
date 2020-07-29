from sklearn.neighbors import KNeighborsRegressor
#ref: https://scikit-learn.org/stable/auto_examples/linear_model/plot_ard.html#sphx-glr-auto-examples-linear-model-plot-ard-py

# 创建一个weights是‘uniform'，一个是'distance'的最近邻回归
def make_nearest_neighbors_regression(X, y):
    knnrs = []
    for i, weights in enumerate(['uniform', 'distance']):
        knnr = KNeighborsRegressor(n_neighbors=5, weights=weights)
        knnr.fit(X, y)
        knnrs.append(knnr)
    return knnrs

if __name__ == "__main__":
    import numpy as np
    X = np.sort(5 * np.random.rand(40, 1), axis=0)
    T = np.linspace(0, 5, 500)[:, np.newaxis]
    y = np.sin(X).ravel()
    import matplotlib.pyplot as plt
    for i, knnr in enumerate(make_nearest_neighbors_regression(X, y)):
        y_ = knnr.predict(T)
        plt.subplot(2, 1, i + 1)
        plt.scatter(X, y, color='darkorange', label='data')
        plt.plot(T, y_, color='navy', label='prediction')
        plt.axis('tight')
    plt.tight_layout()
    plt.show()