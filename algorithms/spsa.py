"""
Simultaneous Perturbation Stochastic Approximation
Author: Jon Eisen
License: MIT

This code defines runs SPSA using iterators.
A quick intro to iterators:
Iterators are like arrays except that we don't store the whole array, we just
store how to get to the next element. In this way, we can create infinite
iterators. In python, iterators can act very similar to arrays.

numpy (a number processing library) is not used here so that pypy (an alternate
python implementation which is faster) can be used.
"""

import random
from abc import *
from itertools import count


# A simple function that returns its argument
def identity(x):
    return x


def SPSA(y, t0, a, c, delta, constraint=identity):
    """
    Creates an Simultaneous Perturbation Stochastic Approximation iterator.
    y - a function of theta that returns a scalar
    t0 - the starting value of theta
    a - an iterable of a_k values
    c - an iterable of c_k values
    delta - a function of no parameters which creates the delta vector
    constraint - a function of theta that returns theta
    """
    theta = t0

    # Pull off the ak and ck values forever
    for ak, ck in zip(a, c):
        # Get estimated gradient
        gk = estimate_gk(y, theta, delta, ck)

        # Adjust theta using SA
        theta = [t - ak * gkk for t, gkk in zip(theta, gk)]

        # Constrain
        theta = constraint(theta)

        yield theta  # This makes this function become an iterator


def estimate_gk(y, theta, delta, ck):
    """Helper function to estimate gk from SPSA"""
    # Generate Delta vector
    delta_k = delta()

    # Get the two perturbed values of theta
    # list comprehensions like this are quite nice
    ta = [t + ck * dk for t, dk in zip(theta, delta_k)]
    tb = [t - ck * dk for t, dk in zip(theta, delta_k)]

    # Calculate g_k(theta_k)
    ya, yb = y(ta), y(tb)
    gk = [(ya - yb) / (2 * ck * dk) for dk in delta_k]

    return gk


def standard_ak(a, A, alpha):
    """Create a generator for values of a_k in the standard form."""
    # Parentheses makes this an iterator comprehension
    # count() is an infinite iterator as 0, 1, 2, ...
    return (a / (k + 1 + A) ** alpha for k in count())


def standard_ck(c, gamma):
    """Create a generator for values of c_k in the standard form."""
    return (c / (k + 1) ** gamma for k in count())


class Bernoulli:
    """
    Bernoulli Perturbation distributions.
    p is the dimension
    +/- r are the alternate values
    """

    def __init__(self, r=1, p=2):
        self.p = p
        self.r = r

    def __call__(self):
        return [random.choice((-self.r, self.r)) for _ in range(self.p)]


class LossFunction(ABC):
    """ A base class for loss functions which defines y as L+epsilon """

    @abstractmethod
    def L(self, theta):
        raise NotImplementedError

    @abstractmethod
    def epsilon(self, theta):
        raise NotImplementedError

    def y(self, theta):
        return self.L(theta) + self.epsilon(theta)
