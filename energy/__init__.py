from abc import *
from typing import Tuple

import numpy as np

import structure
from infrastructure import Loggable


class EnergyTerm(ABC, Loggable):
    @abstractmethod
    def __init__(self):
        self.logger = Loggable.get_default_logger()

    def E(self, *atoms: Tuple[structure.Atom]) -> float:
        raise NotImplementedError


class Bond(EnergyTerm):
    def __init__(self, kb: float, r0: float):
        EnergyTerm.__init__(self)
        #  kJ / mol nm
        self.kb = kb
        #  nm
        self.r0 = r0

    def E(self, i: structure.Atom, j: structure.Atom) -> float:
        return self.kb * np.power(np.norm(i.r - j.r) - self.r0, 2)


class Angle(EnergyTerm):
    def __init__(self, kb: float, t0: float):
        EnergyTerm.__init__(self)
        # kJ / mol rad^2
        self.kb = kb
        # rad
        self.t0 = t0

    def E(self, i: structure.Atom, j: structure.Atom, k: structure.Atom) -> float:
        return self.kb * np.power(structure.Atom.angle(i, j, k) - self.t0, 2)


class Fourier(EnergyTerm):
    def __init__(self, f: Tuple[float, ...]):
        # kJ / mol
        EnergyTerm.__init__(self)
        self.f = f

    def E(self, i: structure.Atom, j: structure.Atom, k: structure.Atom, l: structure.Atom) -> float:
        e = 0.0
        phi = structure.Atom.dihedral(i, j, k, l)

        for i, f in enumerate(self.f):
            # check the actual function definition
            e += f * (1 + np.cos((1 + i) * phi))

        return e


class RyckaertBellemans(EnergyTerm):
    def __init__(self, *c: Tuple[float, ...]):
        # kJ / mol
        EnergyTerm.__init__(self)
        self.c = c

    def E(self, i: structure.Atom, j: structure.Atom, k: structure.Atom, l: structure.Atom) -> float:
        e = 0.0
        phi = structure.Atom.dihedral(i, j, k, l)

        for i, c in enumerate(self.c):
            # check the actual function definition
            e += (-1 if i % 2 else 1) * np.pow(np.cos(phi), i + 1)

        return e
