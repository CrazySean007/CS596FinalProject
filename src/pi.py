from mpi4py import MPI
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def calculate_part(start, step):

    sum = 0.0
    flag = 1
    for i in range(0, step):
        if (start % 2 != 0):
            flag = -1
        else:
            flag = 1
        sum += flag * (1 / (2 * start + 1))
        start += 1
    print(start, " result: ", sum);
    return sum


N = 1024 * 1024 * 64
step = N // size
start = rank * step
t0 = time.time()
value = calculate_part(start, step)
result = 0.0
if rank == 0:
    result += value
    for i in range(1, size):
        value = comm.recv(source=i, tag=0)
        result += value
    t1 = time.time()
    print('PI is : ', result * 4)
    print('time cost is', t1 - t0, 's')

else:
    comm.send(value, dest=0, tag=0)