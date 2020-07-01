import numpy as np


def add_intercept(x):
    """Adds a column of 1's to the non-empty numpy.ndarray x.
    Args:
    x: has to be an numpy.ndarray, a vector of dimension m * 1.
    Returns:
    X as a numpy.ndarray, a vector of dimension m * 2.
    None if x is not a numpy.ndarray.
    None if x is a empty numpy.ndarray.
    Raises:
    This function should not raise any Exception.
    """
    if (not isinstance(x, np.ndarray) or len(x) == 0):
        return None
    if (x.ndim == 1):
        x2 = np.array([x])
    else:
        x2 = x.T
    lin = x2.shape[1]
    v1 = np.full((1, lin), 1)
    return np.concatenate((v1.T, x2.T), axis=1).astype(float)


def gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.ndarray,
     without any for loop. The three arrays must have compatible dimensions.
    Args:
    x: has to be a numpy.ndarray, a matrix of dimension m * 1.
    y: has to be a numpy.ndarray, a vector of dimension m * 1.
    theta: has to be a numpy.ndarray, a 2 * 1 vector.
    Returns:
    The gradient as a numpy.ndarray, a vector of dimension 2 * 1.
    None if x, y, or theta is an empty numpy.ndarray.
    None if x, y and theta do not have compatible dimensions.
    Raises:
    This function should not raise any Exception.
    """
    if (not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray)
            or not isinstance(theta, np.ndarray) or theta.size != 2
            or x.size != y.size or x.ndim != 1 or y.ndim != 1
            or theta.ndim != 1):
        return None
    else:
        x2 = add_intercept(x)
        return ((1 / y.size) * x2.T.dot(x2.dot(theta) - y))


if __name__ == '__main__':
    x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733])
    y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554])
    # Example 0:
    theta1 = np.array([2, 0.7])
    print(gradient(x, y, theta1))
    # Example 1:
    theta2 = np.array([1, -0.4])
    print(gradient(x, y, theta2))
