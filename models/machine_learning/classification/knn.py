from sklearn.neighbors import KNeighborsClassifier

import sys
sys.path.append("C:/Users/KyoDante/Desktop/Torch_Signal_package")

from models.machine_learning.metrics.classification_metrics import get_classification_metrics
def make_knn(X_train, X_test, y_train, y_test,):
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    get_classification_metrics(y_pred, y_test)
    return model

if __name__ == "__main__":
    import numpy as np
    X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1],[-1, -1], [2, 1]])
    X_test = np.array([ [-2, -1], [1, 1],])
    y = np.array([1, 1, 2, 2, 1, 2])
    y_test = np.array([ 1, 2,])
    make_knn(X, X_test, y, y_test)