from mpi4py import MPI
import numpy as np

from ..utils import transfer_data, gather_data
from ..maptypes.raster import Raster 

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

Number = (int, float)


def add(x1, x2):
    return binary_op(x1, x2, np.add)
def sub(x1, x2):
    return binary_op(x1, x2, np.subtract)
def mul(x1, x2):
    return binary_op(x1, x2, np.multiply)
def div(x1, x2):
    return binary_op(x1, x2, np.divide)

def binary_op(x1, x2, op):
    x_size = 0
    y_size = 0
    nodata = None
    driver = None
    georef = None
    proj = None
    if isinstance(x1, Number) and isinstance(x2, Number):
        return op(x1, x2);
    else:
        if isinstance(x1, Raster):
            if rank == 0:
                (x_size, y_size) = x1.x_size, x1.y_size
                (nodata, driver, georef, proj) = x1.get_geo_info()
            x1 = transfer_data(x1.data)
        if isinstance(x2, Raster):
            if rank == 0:
                (x_size, y_size) = x2.x_size, x2.y_size
                (nodata, driver, georef, proj) = x2.get_geo_info()
            x2 = transfer_data(x2.data)
        proc_result = op(x1, x2)
        proc_data = gather_data(proc_result, x_size, y_size)
        comm.Barrier()
    return Raster(None, proc_data, nodata, driver, georef, proj)

    
