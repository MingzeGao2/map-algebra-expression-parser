from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def get_proc_dimensions(x_size, y_size):
    displacement = (0, )
    proc_data_size = ()
    
    normal_y_size = y_size/size
    last_y_size = y_size - normal_y_size * (size-1)
    y_size_list = [normal_y_size]*(size-1)
    y_size_list.append(last_y_size)
    proc_x_size = x_size

    for i in range(size):
        if i == size - 1:
            proc_data_size = proc_data_size + (proc_x_size*last_y_size, )
        else:
            displacement = displacement +  (displacement[-1] + normal_y_size*proc_x_size, )
            proc_data_size = proc_data_size + (proc_x_size*normal_y_size, )

    return (proc_x_size, y_size_list, proc_data_size, displacement)


def transfer_data(data):
    displacement = (0,)
    proc_data_size = ()
    proc_x_size = 0
    proc_y_size = 0
    y_size_list = []

    if rank == 0:
        (proc_x_size, y_size_list, proc_data_size, displacement) = get_proc_dimensions(data.shape[1], data.shape[0])
    proc_y_size = comm.scatter(y_size_list, root=0)
    proc_x_size = comm.bcast(proc_x_size, root=0)
    comm.Barrier()
    proc_data = np.zeros((proc_y_size, proc_x_size), dtype=np.float32)
    
    comm.Scatterv([data, proc_data_size, displacement, MPI.FLOAT], proc_data)
    comm.Barrier()
    return proc_data 

def gather_data(proc_data, x_size, y_size):
    # print "gather data"
    displacement = (0, )
    proc_data_size = ()
    if rank == 0:
        (_, _, proc_data_size, displacement) = get_proc_dimensions(x_size, y_size)
        data = np.zeros((y_size, x_size), dtype=np.float32)
    else:
        data = None
    comm.Gatherv(proc_data, [data, proc_data_size, displacement, MPI.FLOAT])
    comm.Barrier()
    if rank == 0:
        return data
    else:
        return None
