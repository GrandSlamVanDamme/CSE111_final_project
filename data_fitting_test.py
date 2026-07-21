import pytest as pyt  # production by Quincy Jones
import error_analysis as ea
import plots_and_tables as pat
import fitting_functions as ff
from main import X, Y, N, FUNC_TYPES, FUNC_TYPE

import numpy as np


def test_functionator():
    assert ff.functionator(1, 1, "exponential", 70)[2] == 1
    assert ff.functionator(X, Y, "polynomial", 0)[3] == "zeroth"
    assert ff.functionator(X, Y, "polynomial", 7)[3] == "7th degree"
    assert ff.functionator(X, Y, "power", 5)[2] == 1


def test_least_squerror():
    assert ea.leastsquerror([1, 1, 1], [1, 1, 1]) == [0, 0, 0]
    assert ea.leastsquerror([1, 1, 1], [0, 0, 0]) == [1, 2, 3]


def test_resid():
    assert ea.residuals(Y, np.log(Y), "exponential") == len(Y) * [pyt.approx(0)]
    assert ea.residuals(Y, Y) == len(Y) * [pyt.approx(0)]


def test_x_list():
    assert ff.x_list(X)[0] == 1
    assert ff.x_list(X)[-1] == 18 + np.mean(X) / 1000


def test_temp_finder():
    for func in FUNC_TYPES:
        assert ea.temp_finder(X, Y, N, FUNC_TYPE, x=1) > 0
        ea.temp_finder(X, Y, N, FUNC_TYPE, x=1)
        assert ea.temp_finder(X, Y, N, FUNC_TYPE, x=18) > 0
        ea.temp_finder(X, Y, N, FUNC_TYPE, x=18)


pyt.main(["-v", "--tb=line", "-rN", __file__])
