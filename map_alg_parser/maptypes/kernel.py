import numpy as np
import csv

from ..data import read_kernel

class Kernel:
    Kernel = None
    x_size = None
    y_size = None
    def __init__(self, name):
        self.kernel, self.x_size, self.y_size = read_kernel(name)


