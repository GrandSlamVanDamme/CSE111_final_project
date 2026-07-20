"""
This file holds functions that plot and tabulate the
different error-optimized fits
(least squares, Chebyshev, absolute deviation)
of different models (linear, exponential, etc)
which are generated from the functions in fitting_functions.py
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy as sci
import pandas as pd
# import IPython.display as ipd

from fitting_functions import *
from error_analysis import *


def error_compare_table(X, Y, n):
    table_list = []

    for func in FUNC_TYPES:
        f = functionator(X, Y, func, n)
        X, Y, k, func_type = f[0:5]

        entry = error_analyzer(X, Y, k)
        table_list.append(entry)

    table_du_fromage = pd.DataFrame(
        table_list,
        index=pd.MultiIndex.from_product(
            [["Fit Type"], ["linear", poly_degree(n), "power", "exponential"]]
        ),
        columns=pd.MultiIndex.from_product(
            [
                ["Optimization Type"],
                ["Least Squares", "Absolute Deviation", "Chebyshev"],
            ]
        ),
    )

    # ipd.display(table_du_fromage)

    # ipd.Markdown(table_du_fromage.to_markdown(index=False))

    return table_du_fromage


def coeff_table(X, Y, func_type, n):
    """
    Takes X, Y, polynomial degree n.
    Grabs fits and returns their parameters
    for different func_types in a table
    format.
    """

    # for func in func_types:
    table_list = []

    f = functionator(X, Y, func_type, n)
    X, Y, n, func_type = f[0:5]

    LS2_coeffs = LS2_fit(X, Y, n)
    cheb_coeffs = chebyshevify(X, Y, n)
    abs_dev_coeffs = absdev_fit(X, Y, n)

    coeffs_list = [LS2_coeffs, cheb_coeffs, abs_dev_coeffs]

    c_list = [f"c{c}" for c in range(n + 1)]

    for coeff in coeffs_list:
        entry = coeff
        table_list.append(entry)

    table_du_fromage = pd.DataFrame(
        table_list,
        index=pd.MultiIndex.from_product(
            [
                ["Optimization Type"],
                ["Least Squares", "Absolute Deviation", "Chebyshev"],
            ]
        ),
        columns=pd.MultiIndex.from_product([["Function Parameters"], c_list]),
    )

    # ipd.display(table_du_fromage)

    # ipd.Markdown(table_du_fromage.to_markdown(index=False))

    return table_du_fromage


def plotter(X, Y, func_type, n, font_size=12):
    """
    Plots different fit functions for a given data fit type (linear, poly, etc)
    """
    X, Y, n, func_type = functionator(X, Y, func_type, n)[0:4]

    exes = fitter_happier_better(X, Y, n)[0]
    LS2 = fitter_happier_better(X, Y, n)[1]
    cheb = fitter_happier_better(X, Y, n)[2]
    absdev = fitter_happier_better(X, Y, n)[3]

    # Labels so I don't have to type them thrice

    xlab = "Hours passed since Reactor Shutdown"
    ylab = "Reactor temperature in $^{\\circ} C$"

    fig1 = plt.figure(figsize=(7, 20), layout="tight")
    # fig1.suptitle(f"Below are the optimizations for a {func_type} fit")

    plt.rcParams["font.size"] = font_size
    # plt.rcParams["axes.labelpad"] = 10

    dist = 40
    pointsize = 40
    point_color = "black"

    ax1 = plt.subplot(3, 1, 1)
    ax1.scatter(X, Y, s=pointsize, c=point_color)
    ax1.plot(exes, LS2, color="r")
    ax1.set_xlabel(xlab)
    ax1.set_ylabel(ylab, rotation=55, labelpad=(dist), fontsize=12)
    ax1.set_title("Least-Squares Fit")

    ax2 = plt.subplot(3, 1, 2)
    ax2.scatter(X, Y, s=pointsize, c=point_color)
    ax2.plot(exes, cheb, color="y")
    ax2.set_xlabel(xlab)
    ax2.set_ylabel(ylab, rotation=55, labelpad=(dist), fontsize=12)
    ax2.set_title("Chebyshev Fit")

    ax3 = plt.subplot(3, 1, 3)
    plt.scatter(X, Y, s=pointsize, c=point_color)
    plt.plot(exes, absdev, color="b")
    ax3.set_xlabel(xlab)
    ax3.set_ylabel(ylab, rotation=55, labelpad=(dist), fontsize=12)
    ax3.set_title("Absolute Deviation Fit")
