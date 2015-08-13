from numpy import max as npmax
from mpi4py import MPI
from ..maptypes.raster import Raster

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def general_max(*args):
    if rank == 0:
        if len(args)==1:
            if isinstance(args[0],Raster) and args[0].data is not None:
                return npmax(args[0].data)
        else:
            return max(*args)
    else:
        return 0
