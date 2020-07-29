import numpy as np
np.random.seed(0)
from sklearn.model_selection import train_test_split
def make_dataset(data, labels):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)
    return X_train, X_test, y_train, y_test