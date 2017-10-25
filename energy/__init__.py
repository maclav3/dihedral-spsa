from abc import *

import numpy as np

import structure
from infrastructure import Loggable


class EnergyTerm(ABCMeta, Loggable):
    @abstractmethod
    def __init__(self):
        self.logger = Loggable.get_default_logger()

    @abstractmethod
    def E(self, *atoms):
        raise NotImplementedError


class Bond(EnergyTerm):
    def __init__(self, kb, r0):
        EnergyTerm.__init__(self)
        #  kJ / mol nm
        self.kb = kb
        #  nm
        self.r0 = r0

    def E(self, i, j):
        return self.kb * np.power(np.norm(i.r - j.r) - self.r0, 2)


class Angle(EnergyTerm):
    def __init__(self, kb, t0):
        EnergyTerm.__init__(self)
        # kJ / mol rad^2
        self.kb = kb
        # rad
        self.t0 = t0


def E(self, i, j, k):
    return self.kb * np.power(structure.Atom.angle(i, j, k) - self.t0, 2)


class Fourier(EnergyTerm):
    def __init__(self, f):
        EnergyTerm.__init__(self)
        # kJ / mol
        self.f = f

    def E(self, i, j, k, l):
        e = 0.0
        phi = structure.Atom.dihedral(i, j, k, l)

        for i, f in enumerate(self.f):
            # check the actual function definition
            e += f * (1 + np.cos((1 + i) * phi))

        return e


class RyckaertBellemans(EnergyTerm):
    def __init__(self, *c):
        EnergyTerm.__init__(self)
        # kJ / mol
        self.c = c

    def E(self, i, j, k, l):
        e = 0.0

        phi = structure.Atom.dihedral(i, j, k, l)

        for i, c in enumerate(self.c):
            # check the actual function definition
            e += (-1 if i % 2 else 1) * np.pow(np.cos(phi), i + 1)

        return e
