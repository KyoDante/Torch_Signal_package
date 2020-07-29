from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier

import sys
sys.path.append("C:/Users/KyoDante/Desktop/Torch_Signal_package")

# make_pipeline可以像pytorch的transform一样搭建一个必经流程
# StandardScaler完成z标准化 (x-μ) / δ
# SVC是C-Support Vector Classification.
# 实现是基于libsvm的. 拟合时间和样本数成二次方。 
# 因此大数据集下，官方推荐使用sklearn.svm.LinearSVC
# 或者 sklearn.linear_model.SGDClassifier 
# possibly after a sklearn.kernel_approximation.Nystroem transformer.


from models.machine_learning.metrics.classification_metrics import get_classification_metrics
def make_svm(X_train, X_test, y_train, y_test, model_type='SVC'):
    
    model = None
    
    if model_type == 'SVC':
        model = SVC(gamma='auto')
    elif model_type == 'LinearSVC':
        model = LinearSVC()
    elif model_type == 'SGDClassifier':
        model = SGDClassifier()

    # clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    # clf.fit(X_train, y_train)

    # y_pred = clf.predict(X_test)

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
    svm_model = make_svm(X, X_test, y, y_test)

    