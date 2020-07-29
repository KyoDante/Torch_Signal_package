from sklearn.ensemble import RandomForestClassifier

import sys
sys.path.append("C:/Users/KyoDante/Desktop/Torch_Signal_package")

from models.machine_learning.metrics.classification_metrics import get_classification_metrics
def make_randomforest(X_train, X_test, y_train, y_test,):
    model = RandomForestClassifier(max_depth=2, random_state=0)
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
    make_randomforest(X, X_test, y, y_test)