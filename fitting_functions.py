"""
This file contains functions to fit nuclear reactor cooling data
to different models (linear, exponential, etc)
and error optimizations of those models.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy as sci
import pandas as pd


def poly_degree(n):
    """
    takes a whole number n and returns a string for labeling
    a polynomial of degree n.
    """

    degree_list = ["zeroth", "linear", "quadratic", "cubic", "quartic", "quintic"]

    k = 0
    while k < 95:
        degree_list.append(f"{k + 6}th degree")
        k += 1

    return degree_list[n]


def functionator(X, Y, func_type, n):
    """
    Takes X, Y, and func type.
    Modifies X and Y if needed
    returns X, Y, n, func_type as list
    """
    fit_type = func_type
    k = n
    try:
        if fit_type == "polynomial":
            fit_type = poly_degree(n)
        elif fit_type == "exponential":
            Y = np.log(Y)
            k = 1
        elif fit_type == "power":
            Y = np.log(Y)
            X = np.log(X)
            k = 1
        elif fit_type == "linear":
            k = 1
    except TypeError:  # if n is not an integer
        print("Request cannot be completed, defaulting to rat")
        import matplotlib.image as mpimg

        img = mpimg.imread("fat_rats/Joanna_Servaes_wikimedia_commons.webp")
        plt.imshow(img)
        plt.axis("off")
        plt.imshow(img)
        plt.axis("off")
        plt.imshow(img)
        plt.axis("off")
        plt.show()

        fit_type = None
        return

    return [X, Y, k, fit_type]


def objectivist(X, Y, n):
    """
    Linear programming optimization.
    Template: https://byui.instructure.com/courses/409534/pages/w07-tuesday-lesson-plans-2?module_item_id=4500264
    Takes two arrays, x = [data 1] and Y = [data 2].
    f(x) = c_n*x^n + c_(n-1)x^(n-1)+...c_0x^0 with n parameters
    returns objective function f
    """

    mat_x = []
    mat_y = []
    x_list = []

    for Xi, Yi in zip(X, Y):
        i = n

        rightrow = []
        leftrow = []
        objective = []
        bounds = []

        while i > -1:
            bounds.append((None, None))
            objective.append(0)
            rightrow.append(Xi**i)
            leftrow.append(-(Xi**i))
            x_list.append(Xi**i)

            i -= 1

        rightrow.append(-1)
        leftrow.append(-1)
        mat_x.append(rightrow)
        mat_x.append(leftrow)
        mat_y.append(Yi)
        mat_y.append(-Yi)

    bounds.append((0, None))
    objective.append(1)

    barry = [objective, mat_x, mat_y, bounds, x_list]
    return barry


def chebyshevify(X, Y, n):
    """
    Takes two arrays, x = [data 1] and Y = [data 2].
    Using Chebyshev criteria, fits model function
    f(x) = c_n*x^n + c_(n-1)x^(n-1)+...c_0x^0 with n parameters
    such that (max|Yi-Fi|, i ϵ NN) is minimized.
    returns F(x) as an array of coefficients, the last being the max error E
    """

    barry = objectivist(X, Y, n)

    result = sci.optimize.linprog(
        barry[0], A_ub=barry[1], b_ub=barry[2], bounds=barry[3], method="highs"
    )

    result = np.delete(result.x, -1)

    return result


def absdev_fit(X, Y, n):
    """
    takes X data and makes a list of powers
    (X^n, X^n-1...),
    dots it with undefined params to whip up
    a cost function that absdev_fit
    will minimize.

    https://stackoverflow.com/questions/51883058/l1-norm-instead-of-l2-norm-for-cost-function-in-regression-model

    used for troubleshooting and template.
    """

    def cost_func(params):
        # sympy removal courtesy of gemini thread listed up top
        # SciPy will plug the current numerical guess into 'params'
        f = np.polyval(params, X)
        # Return the L1 norm (sum of absolute deviations)
        return np.sum(np.abs(Y - f))

    guess = (n + 1) * [1.0]

    result = sci.optimize.minimize(cost_func, guess, method="Nelder-Mead")

    return result.x


def Ls2_fit(X, Y, n):
    """
    Takes two arrays, x = [data 1] and Y = [data 2].
    fits model function
    f(x) = c_n*x^n + c_(n-1)x^(n-1)+...c_0x^0 with n parameters
    such that (sum|Yi-Fi|^2, i ϵ NN) is minimized.
    returns f(x) as an array
    """
    Ls2 = np.polyfit(X, Y, n)

    return Ls2


def x_list(X):
    """
    takes X data, returns linspaced list of X-powers
    from X^n to X^0.
    """
    X = np.linspace(X[0], X[-1] + np.mean(X) / 1000, 1000)

    return X


def fitter_happier_better(X, Y, n):
    """
    Takes X, Y, and polynomial degree n.
    Grabs coefficients from the fit generators
    and returns arrays of actual function output.
    """
    LS2_coeffs = Ls2_fit(X, Y, n)
    cheb_coeffs = chebyshevify(X, Y, n)
    abs_dev_coeffs = absdev_fit(X, Y, n)

    exes = x_list(X)

    Ls2 = np.polyval(LS2_coeffs, exes)
    cheb = np.polyval(cheb_coeffs, exes)
    absdev = np.polyval(abs_dev_coeffs, exes)

    return [exes, Ls2, cheb, absdev]
