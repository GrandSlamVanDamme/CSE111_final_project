"""
Functions to perform error analysis on fitting functions using
three common optimization methods:

least squares (minimize the sum of squared residuals)
absolute deviation (minimize the sum of residuals)
Chebyshev (minimize the maximum residual)
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy as sci
import pandas as pd

from fitting_functions import *


def error_analyzer(X, Y, n, k=-1):
    """
    prints results of least squares, chebyshev, and absolute deviation analyses for
    type of model given by model_name and degree given by n. For error up to an arbitrary point, edit indices
    of the individual error functions. Default value of k is -1.
    """

    LS2 = leastsquerror(Y, np.polyval(LS2_fit(X, Y, n), X))[k]
    AbsDev = absdev_error(Y, np.polyval(absdev_fit(X, Y, n), X))[k]
    Cheby = cheb_error(Y, np.polyval(chebyshevify(X, Y, n), X))

    return [LS2, AbsDev, Cheby]


def leastsquerror(Y, F):
    """
    Takes two lists, Y = [data points] and F(x) = [func_values].
    Computes cumulative variances of a series of residuals
    returns list of variances
    """
    resid = residuals(Y, F)

    squares = []

    squm = []

    for r in resid:
        squares.append(r**2)

        squm.append(sum(squares))

    return squm


def absdev_error(Y, F):
    """
    Takes two lists, Y = [data points] and F(x) = [func_values].
    computes cumulative absolute deviation
    returns list of absolute deviations
    """

    resid = residuals(Y, F)

    normals = []

    absum = []

    for r in resid:
        normals.append(abs(r))

        absum.append(sum(normals))

    return absum


def cheb_error(Y, F):
    """
    Takes two lists, Y = [data points] and F(x) = [func_values].
    returns maximum residual
    """

    cheb = abs(max(residuals(Y, F)))

    return cheb


def residuals(Y, F, func_type="linear"):
    """
    Takes two lists, Y = [data points] and F(x) = [func_values]
    computes their residuals
    returns list of residuals

    """

    resid = []
    if func_type == "exponential" or func_type == "power":
        Y = [np.e**y for y in Y]
    else:
        pass

    for f, y in zip(F, Y):
        resid.append(y - f)

    return resid


def arrayer(x, Y):
    """
    requires numpy
    takes two lists and makes them into arrays
    """

    x_array = np.array(x)
    Y_array = np.array(Y)

    return x_array, Y_array


def temp_finder(X, Y, n):
    """
    Takes X, Y, n then obtains
    user input for time since
    reactor shutdown (Will add this feature later).
    Returns projected temp
    as calculated from Chebyshev-optimized
    exponential fit
    """

    """
    x = float(input("How many hours has it been since the reactor shut down"))
    """

    x = 24
    X, Y, n = functionator(X, Y, "exponential", n)[0:3]

    # LS2_coeffs = LS2_fit(X, Y, n)
    cheb_coeffs = chebyshevify(X, Y, n)
    # abs_dev_coeffs = absdev_fit(X, Y, n)

    # LS2 = np.polyval(LS2_coeffs, x)
    cheb = np.polyval(cheb_coeffs, x)
    # absdev = np.polyval(abs_dev_coeffs, x)

    T = f"After {x} hours, we project an estimated reactor temperature of {cheb:.3f} degrees centigrade."

    return T
