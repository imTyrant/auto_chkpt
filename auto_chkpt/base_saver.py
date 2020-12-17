from abc import ABC, abstractmethod

class BaseSaver(ABC):
    def __init__(self):
        """
        """

    @abstractmethod
    def step(self):
        """
        """

    @abstractmethod
    def save(self):
        """
        """
    
    @abstractmethod
    def load(self):
        """
        """

    @abstractmethod
    def exception(self):
        """
        """
    
    
