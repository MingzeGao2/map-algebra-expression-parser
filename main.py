from mpi4py import MPI
import sys

from map_alg_parser.parser import parse, eval 

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def schemestr(exp):
    "Convert a Python object back into a Schem-readable string."
    if isinstance(exp, list):
        return '(' + ' ' .join(map(schemestr, exp)) + ')'
    else:
        return str(exp)

def repl(prompt='lis.py>'):
    "A prompt-read-eval-print loop."
    parsed_str = None
    while True:
        if (rank == 0):        
            parsed_str = parse(raw_input(prompt))
        comm.Barrier()
        parsed_str = comm.bcast(parsed_str, root = 0)
        comm.Barrier()
        val = eval(parsed_str)
        comm.Barrier()
        if rank == 0 and val is not None:
            print(schemestr(val))

def run_script(script):
    parsed_str = None
    lines = [line.rstrip('\n') for line in open(script)]
    for line in lines:
        if(rank == 0):
            print line
            parsed_str = parse(line)
        comm.Barrier()
        parsed_str = comm.bcast(parsed_str, root=0)
        comm.Barrier()
        val = eval(parsed_str)
        comm.Barrier()
        if rank == 0 and val is not None:
            print "\nresult:"
            print (schemestr(val))
            print "\n"


if __name__ =='__main__':
    try:
        run_script(sys.argv[1])
    except:
        repl()
