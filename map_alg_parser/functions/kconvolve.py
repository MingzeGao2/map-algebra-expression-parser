from mpi4py import MPI
from ..maptypes.raster import Raster

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def kconvolve(raster, kernel):
    if rank == 0:
        print kernel.kernel
