from sklearn import metrics

# ref: https://blog.csdn.net/Yqq19950707/article/details/90169913
def get_regression_metrics(y_pred, y_test):
    
    # 回归方差
    explained_variance = metrics.explained_variance_score(y_test, y_pred, )
    print(f"explained_variance: {explained_variance}")

    # 平均绝对误差
    mean_absolute_error = metrics.mean_absolute_error(y_test, y_pred, )
    print(f"mean_absolute_error: {mean_absolute_error}")

    # 均方差
    mean_squared_error = metrics.mean_squared_error(y_test, y_pred)
    print(f"mean_squared_error: {mean_squared_error}")

    # 中值绝对误差
    median_absolute_error = metrics.median_absolute_error(y_test, y_pred)
    print(f"median_absolute_error: {median_absolute_error}")

    # R平方值
    r2_score = metrics.r2_score(y_test, y_pred)
    print(f"r2_score: {r2_score}")
