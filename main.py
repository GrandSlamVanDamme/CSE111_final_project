#!/usr/bin/env python
# coding: utf-8

# In[6]:


"""
main.py

This file contains the master function for fitting, analysis, and display
of nuclear reactor cooling data originally taken from math 283.
For CSE 111, this file has been broken up, with data displaying functions
moved to plots_&_tables.py and data fitting moved to fitting_functions.py.
author: Corey Silver
original date: 3/6/26 (Yuro Style)
latest revision: 20/7/26

functions require numpy, scipy, matplotlib, and pandas libraries

troubleshooting suggestions (frequently wrong) from
gemini: https://gemini.google.com/share/de2be7b1f7ee
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy as sci
import pandas as pd

import plots_and_tables as pat
import error_analysis as ea
import fitting_functions as ff

X = [1, 2, 4, 8, 12, 18]  # hours from reactor shutdown
Y = [580, 510, 430, 340, 290, 230]  # temperature of reactor in C
N = 3  # polynomial degree.
# Note: Exp and power fits are modeled in linear log-log by functionator().
FUNC_TYPES = ["linear", "polynomial", "power", "exponential"]
FUNC_TYPE = "polynomial"
# Note: FUNC_TYPE can be "linear", "polynomial", "power", or "exponential".


def main():
    """
    The big Huncho Grande Paparoni: fitting, error analysis, plotting.
    """
    pat.error_compare_table(X, Y, N, FUNC_TYPES)
    pat.coeff_table(X, Y, FUNC_TYPE, N)
    pat.plotter(X, Y, FUNC_TYPE, N)


if __name__ == "__main__":
    main()

print(ea.residuals(Y, Y))
print(
    ea.residuals(
        Y,
        np.polyval(
            ff.chebyshevify(
                ff.functionator(X, Y, "exponential", N)[0],
                ff.functionator(X, Y, "exponential", N)[1],
                ff.functionator(X, Y, "exponential", N)[2],
            ),
            ff.x_list(X),
        ),
    )
)
print(
    ff.chebyshevify(
        ff.functionator(X, Y, "exponential", N)[0],
        ff.functionator(X, Y, "exponential", N)[1],
        ff.functionator(X, Y, "exponential", N)[2],
    )
)
