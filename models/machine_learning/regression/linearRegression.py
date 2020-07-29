import numpy as np
from sklearn.linear_model import ARDRegression, LinearRegression

# ref: https://scikit-learn.org/stable/auto_examples/linear_model/plot_ard.html#sphx-glr-auto-examples-linear-model-plot-ard-py

def make_linear(X, y):
    n_samples = np.shape(X)[0]
    n_features = np.shape(X)[1]

    ard = ARDRegression(compute_score=True)
    ard.fit(X, y)

    ols = LinearRegression()
    ols.fit(X, y)

    return ard, ols


if __name__ == "__main__":
    X = np.random.randn(100, 100)
    lambda_ = 4.
    w = np.zeros(100)
    relevant_features = np.random.randint(0, 100, 10)
    from scipy import stats
    for i in relevant_features:
        w[i] = stats.norm.rvs(loc=0, scale=1./ np.sqrt(lambda_))
    alpha_ = 50.
    noise = stats.norm.rvs(loc=0, scale=1. / np.sqrt(alpha_), size=100)

    y = np.dot(X, w) + noise

    ard, ols = make_linear(X, y)

    # #############################################################################
    # Plot the true weights, the estimated weights, the histogram of the
    # weights, and predictions with standard deviations
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 5))
    plt.title("Weights of the model")
    plt.plot(ard.coef_, color='darkblue', linestyle='-', linewidth=2,
            label="ARD estimate")
    plt.plot(ols.coef_, color='yellowgreen', linestyle=':', linewidth=2,
            label="OLS estimate")
    plt.plot(w, color='orange', linestyle='-', linewidth=2, label="Ground truth")
    plt.xlabel("Features")
    plt.ylabel("Values of the weights")
    plt.legend(loc=1)
    plt.show()