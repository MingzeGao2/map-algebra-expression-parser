import os.path
import numpy as np
from mpi4py import MPI

from ..data import *
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class Raster:    
    name = None
    data = None
    nodata = None
    driver = None
    georef= None
    proj = None
    x_size = None
    y_size = None
    def __init__(self, name, data=None, nodata=None, driver=None, georef=None, proj=None):
        "constructor for raster"
        if rank == 0:
            if name is None:        # intermidiate raster has no name 
                self.name = None
                self.data = data
                self.nodata = nodata
                self.driver = driver
                self.georef = georef
                self.proj = proj
                self.x_size = None if data is None  else data.shape[1]
                self.y_size = None if data is None else data.shape[0]
            else:                   # user define empty raster only has name 
                if not os.path.isfile(name):
                    self.name = name
                else:               # raster already in file system
                    self.name = name
                    (self.data, self.x_size, self.y_size,
                     self.nodata, self.driver, self.georef, self.proj) = read_raster(name)
        else:                   # raster in other process other than  root process wil be empty 
            self.name = name

    def __repr__(self):
        return '<%s.%s object at %s>' % (
            self.__class__.__module__,
            self.__class__.__name__,
            hex(id(self))
        )

        
    def __str__(self):
        print self.__repr__()
        if self.data is not None:
            return  str(self.data)
        elif self.name is not None:
            return self.name
        else:
            return ' '
    
    def __abs__(self):
        if rank == 0:
            result = np.absolute(self.data)
        else:
            result = None
        return Raster(None, result, self.nodata, self.driver, self.georef, self.proj)        

    def get_geo_info(self):
        return (self.nodata, self.driver, self.georef, self.proj)

    def write_data(self, data, x_offset, y_offset, nodata,
              x_size, y_size, driver, georef, proj ):
        if rank == 0:
            self.x_size = x_size;
            self.y_size = y_size;
            self.driver = driver
            self.nodata = nodata
            self.data = data
            self.georef = georef
            self.proj = proj
            dataset = create_raster(x_size, y_size, self.name, 
                                    driver, georef, proj)
            write_raster(dataset, data, x_offset, y_offset, nodata)
