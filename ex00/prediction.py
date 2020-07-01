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


def predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty np.ndarray.
    Args:
    x: has to be an np.ndarray, a vector of dimension m * 1.
    theta: has to be an np.ndarray, a vector of dimension 2 * 1.
    Returns:
    y_hat as a np.ndarray, a vector of dimension m * 1.
    None if x or theta are empty np.ndarray.
    None if x or theta dimensions are not appropriate.
    Raises:
    This function should not raise any Exceptions.
    """
    if (not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray)
            or len(x) == 0 or len(theta) != 2):
        return None
    else:
        x = add_intercept(x)
        return x.dot(theta)


if __name__ == '__main__':
    x = np.arange(1, 6)
    # Example 1:
    theta1 = np.array([5, 0])
    print(predict_(x, theta1))
    # Do you remember why y_hat contains only 5's here?
    # Example 2:
    theta2 = np.array([0, 1])
    print(predict_(x, theta2))
    # Do you remember why y_hat == x here?
    # Example 3:
    theta3 = np.array([5, 3])
    print(predict_(x, theta3))
    # Example 4:
    theta4 = np.array([-3, 1])
    print(predict_(x, theta4))
