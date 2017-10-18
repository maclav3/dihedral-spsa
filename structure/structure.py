from abc import *
from typing import List

import numpy as np
from numpy.linalg import norm

_normalize = lambda v: v / norm(v)


class Atom(object):
    def __init__(self, x: float, y: float, z: float):
        self.r = np.array([x, y, z])

    @staticmethod
    def distance(i: 'Atom', j: 'Atom') -> float:
        return norm(i.r - j.r)

    @staticmethod
    def angle(i: 'Atom', j: 'Atom', k: 'Atom') -> float:
        ji = _normalize(j.r - i.r)
        jk = _normalize(j.r - k.r)

        return np.arccos(np.dot(ji, jk))

    @staticmethod
    def dihedral(i: 'Atom', j: 'Atom', k: 'Atom', l: 'Atom') -> float:
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


class Structure(ABC):
    @property
    @abstractmethod
    def atoms(self) -> List[Atom]:
        return self.atoms

    # reads a structure from file at filename and fills atoms
    @abstractmethod
    def __init__(self, filename: str):
        self.atoms = None
        raise NotImplementedError
