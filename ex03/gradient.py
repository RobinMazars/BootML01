import numpy as np
from prediction import predict_


def simple_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.ndarray,
    without any for-loop. The three arrays must have compatible dimensions.
    Args:
    x: has to be an numpy.ndarray, a vector of dimension m * 1.
    y: has to be an numpy.ndarray, a vector of dimension m * 1.
    theta: has to be an numpy.ndarray, a 2 * 1 vector.
    Returns:
    The gradient as a numpy.ndarray, a vector of dimension 2 * 1.
    None if x, y, or theta are empty numpy.ndarray.
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
        h = predict_(x, theta)
        j0 = (1 / y.size) * (h - y)
        j0 = abs(j0.sum())
        j1 = (1 / y.size) * ((h - y) * x)
        j1 = abs(j1.sum())
        return np.array((j0, j1))


if __name__ == '__main__':
    x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733])
    y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554])
    # Example 0:
    theta1 = np.array([2, 0.7])
    print(simple_gradient(x, y, theta1))
    # Example 1:
    theta2 = np.array([1, -0.4])
    print(simple_gradient(x, y, theta2))
