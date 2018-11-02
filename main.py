from math import exp, cos, sin
import matplotlib.pyplot as plt
import numpy


def calculate_error(function):
    """ Wrapper to calculate local errors for approximated values"""

    def wrapper(f, xs, y0, h, y_solution):
        approximated = function(f, xs, y0, h)
        error = get_error(y_solution, approximated)
        return approximated, error

    return wrapper


def func(x, y):
    """y' = f(x,y)"""

    val = -2 * y + 4 * x
    return val


def get_error(ys, ysol):
    """Calculate error"""

    error = []
    for yi_sol, yi_apr in zip(ys, ysol):
        error += [abs(yi_sol - yi_apr)]
    return error


# -1 + e^(-2 x) + 2 x
def exact_solution(x):
    return -1 + exp(-2 * x) + 2 * x


def solution(xs):
    """Calculate exact solution for every x from xs
    (xs - array of x values with some step)"""

    ys = []
    for xi in xs:
        ys += [exact_solution(xi)]
    return ys

"""f - callable function of x and y such that y'=f(x,y),
    xs - array of x-values
    y0 - given ivp value for xs[0]
    h - step
"""

@calculate_error
def euler(f, xs, y0, h):
    ys = [y0]
    for i in range(1, xs.size):
        ys += [ys[i - 1] + h * f(xs[i - 1], ys[i - 1])]
    return ys


@calculate_error
def improved_euler(f, xs, y0, h):
    ys = [y0]
    for i in range(1, xs.size):
        k1 = f(xs[i - 1], ys[i - 1])
        k2 = ys[i - 1] + h * k1
        ys += [ys[i - 1] + h / 2 * (k1 + f(xs[i], k2))]
    return ys


@calculate_error
def runge_khutta(f, xs, y0, h):
    ys = [y0]
    for i in range(1, xs.size):
        k1 = f(xs[i - 1], ys[i - 1])
        k2 = f(xs[i - 1] + h / 2, ys[i - 1] + h * k1 / 2)
        k3 = f(xs[i - 1] + h / 2, ys[i - 1] + h * k2 / 2)
        k4 = f(xs[i - 1] + h, ys[i - 1] + h * k3)
        ys += [ys[i - 1] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)]
    return ys

if __name__ == '__main__':
    euler_, improved_, runge_, exact_ = 'Euler', 'Improved Euler', 'Runge Kutta', 'Exact solution'

    x0 = float(input('Enter x0:'))
    y0 = float(input('Enter y0:'))
    x1 = float(input('Enter xn:'))
    h = float(input('Enter step:'))
    # Build array of x values in range [x0,x1] with step h
    xs = numpy.arange(x0, x1 + h, h)
    ysol = solution(xs)
    tup = (func, xs, y0, h, ysol)
    # draw euler
    y, error1 = list(euler(*tup))
    euler_plot = plt.plot(xs, y, 'b', label=euler_)
    # draw exact solution
    exact_plot = plt.plot(xs, ysol, 'r', label=exact_)
    # draw improved euler
    y, error2 = list(improved_euler(*tup))
    impr_plot = plt.plot(xs, y, 'y', label=improved_)
    # draw runge-khutta
    y, error3 = list(runge_khutta(*tup))
    runge_plot = plt.plot(xs, y, 'g', label=runge_)

    plt.legend()
    plt.show()
    # plot errors
    plt.plot(xs, error1, 'b', label=euler_)
    plt.plot(xs, error2, 'y', label=improved_)
    plt.plot(xs, error3, 'g', label=runge_)
    plt.legend()
    plt.show()
    # plot runge-khutta error with more precision separately
    plt.plot(xs, error3, 'g', label=runge_)
    plt.legend()
    plt.show()
