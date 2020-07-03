import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt


class MyLinearRegression():
    """
    Description:
    My personnal linear regression class to fit like a boss.
    """

    def __init__(self, thetas, alpha=0.0001, n_cycle=100000):
        self.alpha = alpha
        self.n_cycle = n_cycle
        self.thetas = np.array((thetas[0], thetas[1]))

    def fit_(self, x, y):
        step_tolerance = 0.00001
        thetatemp0 = self.thetas[0]
        thetatemp1 = self.thetas[1]

        for i in range(self.n_cycle):
            g = self.gradient(x, y, np.array((thetatemp0, thetatemp1)))
            # print(g)
            thetatemp0b = thetatemp0 - (self.alpha * g[0])
            thetatemp1b = thetatemp1 - (self.alpha * g[1])
            # print(thetatemp1b)
            # print(thetatemp1b)
            # if (thetatemp1 * thetatemp1b) < 0:
            if - step_tolerance < thetatemp0b - thetatemp0 < step_tolerance:
                # print(f" nb of step = {i+1}")
                break
            thetatemp0 = thetatemp0b
            thetatemp1 = thetatemp1b
        self.thetas = np.array((thetatemp0, thetatemp1))

    def predict_(self, x):
        if (not isinstance(x, np.ndarray)
                or len(x) == 0 or len(self.thetas) != 2):
            return None
        else:
            x = self.add_intercept(x)
            return x.dot(self.thetas)

    def cost_elem_(self, x, y):
        if (x.ndim == 1):
            x = (np.array([x])).T
        if (y.ndim == 1):
            y = (np.array([y])).T
        return (1 / (2 * x.size)) * pow((y - x), 2)

    def cost_(self, x, y):
        return self.cost_elem_(x, y).sum()

    def add_intercept(self, x):
        if (not isinstance(x, np.ndarray) or len(x) == 0):
            return None
        if (x.ndim == 1):
            x2 = np.array([x])
        else:
            x2 = x.T
        lin = x2.shape[1]
        v1 = np.full((1, lin), 1)
        return np.concatenate((v1.T, x2.T), axis=1).astype(float)

    def gradient(self, x, y, theta):
        if (not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray)
                or not isinstance(theta, np.ndarray) or theta.size != 2
                or x.size != y.size or x.ndim != 1 or y.ndim != 1
                or theta.ndim != 1):
            return None
        else:
            x2 = self.add_intercept(x)
            # print(x2)
            # print(x2.T)
            return ((1 / y.size) * x2.T.dot(x2.dot(theta) - y))


if __name__ == '__main__':
    x = np.array([12.4956442, 21.5007972,
                  31.5527382, 48.9145838, 57.5088733])
    y = np.array([37.4013816, 36.1473236,
                  45.7655287, 46.6793434, 59.5585554])
    x2 = np.array([[12.4956442], [21.5007972], [
                  31.5527382], [48.9145838], [57.5088733]])
    y2 = np.array([[37.4013816], [36.1473236], [
        45.7655287], [46.6793434], [59.5585554]])
    print("lr1")
    lr1 = MyLinearRegression([2, 0.7])
    print(lr1.predict_(x))
    print(lr1.cost_elem_(lr1.predict_(x), y))
    print(lr1.cost_(lr1.predict_(x), y))
    print("lr2")
    print("true")
    lr2_test = linear_model.LinearRegression()
    lr2_test.fit(x2, y2)
    print(lr2_test.predict(x2))
    print("mine")

    lr2 = MyLinearRegression([0, 0])
    lr2.fit_(x, y)
    print(lr2.thetas)
    p = lr2.predict_(x)
    print(p)
    print(lr2.cost_elem_(lr1.predict_(x), y))
    print(lr2.cost_(lr1.predict_(x), y))
    plt.plot(x, y, "o")
    plt.plot(x, p, "r")
    plt.show()
