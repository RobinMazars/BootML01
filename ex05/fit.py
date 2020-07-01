import numpy as np
from prediction import predict_
from vec_gradient import gradient
from scipy import stats
import matplotlib.pyplot as plt


def fit_(x, y, theta, alpha, max_iter):
    """
    Description:
    Fits the model to the training dataset contained in x and y.
    Args:
    x: has to be a np.ndarray, a vector of dim m * 1: (nb of training ex, 1).
    y: has to be a np.ndarray, a vector of dim m * 1: (nb of training ex, 1).
    theta: has to be a np.ndarray, a vector of dim 2 * 1.
    alpha: has to be a float, the learning rate
    max_iter: has to be an int, the nb of iter done during the gradient descent
    Returns:
    new_theta: np.ndarray, a vector of dim 2 * 1.
    None if there is a matching dim problem.
    Raises:
    This function should not raise any Exception.
    """
    step_tolerance = 0.00001

    thetatemp0 = theta[0]
    thetatemp1 = theta[1]
    for i in range(max_iter):
        g = gradient(x, y, np.array((thetatemp0, thetatemp1)))
        # print(g)
        thetatemp0b = thetatemp0 - (alpha * g[0])
        thetatemp1b = thetatemp1 - (alpha * g[1])
        # print(thetatemp1b)
        # print(thetatemp1b)
        # if (thetatemp1 * thetatemp1b) < 0:
        if - step_tolerance < thetatemp0b - thetatemp0 < step_tolerance:
            print(f" nb of step = {i+1}")
            break
        thetatemp0 = thetatemp0b
        thetatemp1 = thetatemp1b
    return np.array((thetatemp0, thetatemp1))


x = np.array([12.4956442, 21.5007972,
              31.5527382, 48.9145838, 57.5088733])
y = np.array([37.4013816, 36.1473236,
              45.7655287, 46.6793434, 59.5585554])
theta = np.array([1, 1])
# Example 0:
theta1 = fit_(x, y, theta, alpha=9e-4, max_iter=1500000)
print(theta1)
# Example 1:
p = predict_(x, theta1)
print(p)
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
plt.plot(x, y, 'b', label='original data')
plt.plot(x, intercept + slope * x, 'r', label='fitted line')
plt.plot(x, p, 'y', label='my line2')
plt.legend()
plt.show()
