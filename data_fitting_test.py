import pytest as pyt  # production by Quincy Jones
import error_analysis as ea
import plots_and_tables as pat
import fitting_functions as ff
from main import X, Y, N, FUNC_TYPES, FUNC_TYPE

def test_functionator():
    assert ff.functionator(1, 1, "exponential", 70) == [1, 0, 1, "exponential"]
    assert ff.functionator(X, Y, "polynomial", 0) == [X, Y, 3, "cubic"]
    assert ff.functionator(X, Y, "power", 5) == [X, Y, 1, "power"]

def test_least_squerror():


def test_resid():

def test_x_list():

def test_temp_finder():

