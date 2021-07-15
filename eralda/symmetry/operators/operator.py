from abc import ABC, abstractmethod


class SymmetryOperator(ABC):
    def __init__(self):
        pass

    @property
    @abstractmethod
    def matrices(self):
        raise NotImplementedError

