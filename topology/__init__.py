from abc import *


class Topology(ABCMeta):
    @abstractmethod
    def __init__(self):
        self.bonds = None
        self.angles = None
        self.dihedrals = None
