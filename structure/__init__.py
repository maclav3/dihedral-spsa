from abc import *

import numpy as np
from numpy.linalg import norm

_normalize = lambda v: v / norm(v)


class Atom(ABCMeta):
    def __init__(self, x, y, z):
        self.r = np.array([x, y, z])

    @staticmethod
    def distance(i, j):
        return norm(i.r - j.r)

    @staticmethod
    def angle(i, j, k):
        ji = _normalize(j.r - i.r)
        jk = _normalize(j.r - k.r)

        return np.arccos(np.dot(ji, jk))

    @staticmethod
    def dihedral(i, j, k, l):
        # https://stackoverflow.com/questions/20305272/
        # dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python
        ji = j.r - i.r
        kj = _normalize(k.r - j.r)
        lk = l.r - k.r

        # vector rejections
        # v = projection of ji onto plane perpendicular to kj
        #   = ji minus component that aligns with kj
        # w = projection of lk onto plane perpendicular to kj
        #   = lk minus component that aligns with kj
        v = ji - np.dot(ji, kj) * kj
        w = lk - np.dot(lk, kj) * kj

        # angle between v and w in a plane is the torsion angle
        # v and w may not be normalized but that's fine since tan is y/x
        x = np.dot(v, w)
        y = np.dot(np.cross(kj, v), w)

        return np.arctan2(y, x)


class Structure(ABCMeta):
    @property
    @abstractmethod
    def atoms(self):
        return self.atoms

    # reads a structure from file at filename and fills atoms
    @abstractmethod
    def __init__(self, filename):
        self._atoms = None
        raise NotImplementedError
