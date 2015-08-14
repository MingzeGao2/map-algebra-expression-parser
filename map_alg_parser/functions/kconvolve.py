from mpi4py import MPI
import numpy as np

from ..utils import transfer_data, gather_data
from ..utils import get_proc_dimensions_kconvolve, get_proc_dimensions
from ..maptypes.raster import Raster
from ..maptypes.kernel import Kernel

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def kconvolve(raster, kernel):
    x_size = 0
    y_size = 0
    nodata = None
    driver = None
    georef = None
    proj = None

    displacement = (0,)
    proc_data_size = ()
    proc_x_size = 0
    proc_y_size = 0
    y_size_list = []

    if rank == 0:
        (x_size, y_size) = raster.x_size, raster.y_size
        (nodata, driver, georefj, proj) = raster.get_geo_info()
        (proc_x_size, y_size_list, proc_data_size, displacement)\
            = get_proc_dimensions_kconvolve(x_size, y_size, kernel.y_size)
    comm.Barrier()
    proc_data = transfer_data(raster.data, proc_x_size, y_size_list, 
                              proc_data_size, displacement)
    proc_result = kernel_process(proc_data, kernel.kernel)
    comm.Barrier()
    if rank == 0:
        (_, _, proc_data_size, displacement)\
            = get_proc_dimensions(x_size, y_size)
    result_data = gather_data(x_size, y_size, proc_result, 
                              proc_data_size, displacement)
    comm.Barrier()
    return Raster(None, result_data, nodata, driver, georef, proj)

def kernel_process(data, kernel):
    blocks = get_block(data, kernel.shape[1],kernel.shape[0])
    result = np.zeros(blocks[0].shape, dtype=np.float32)
    flat_kernel = kernel.reshape(kernel.shape[0]*kernel.shape[1])
    for i in range(len(blocks)):
        result = np.add(result, np.multiply(flat_kernel[i], blocks[i]))
    if rank == 0 and size > 1:
        result = np.vstack([np.zeros((kernel.shape[0]/2, result.shape[1]),dtype=np.float32), result])
    if rank == size -1 and size > 1:
        result = np.vstack([result, np.zeros((kernel.shape[0]/2, result.shape[1]),dtype=np.float32)])
    result = np.hstack([np.zeros((result.shape[0],kernel.shape[1]/2),dtype=np.float32),\
                        result, np.zeros((result.shape[0],kernel.shape[1]/2),dtype=np.float32)])
    return result

def get_block(data, kernel_x_size, kernel_y_size):
    bar = []
    overlap_x = kernel_x_size/2
    overlap_y = kernel_y_size/2
    for y in range (kernel_y_size):
        for x in range(kernel_x_size):
            y_bound = None if (overlap_y*2==y) else -(overlap_y*2-y)
            x_bound = None if (overlap_x*2==x) else -(overlap_x*2-x)
            bar.append(data[y:y_bound, x:x_bound])
    return bar
