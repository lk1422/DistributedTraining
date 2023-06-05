import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Server
import Aux
from asyncio import StreamReader, StreamWriter
import asyncio

CLOSE_CONNECTION = {'KILL': True}

class Worker:
    def __init__(self, address : tuple , reader : StreamReader, 
                 writer :StreamWriter ):
        self.address = tuple
        self.reader = reader
        self.writer = writer

async def setup_host(num_connections : int):
    await Server.get_connections(num_connections)
    workers = Server.OPEN_SOCKETS

    new_worker_list = []
    for worker in workers:
        worker_obj = Worker(worker[0], worker[1], worker[2])
        new_worker_list.append(worker_obj)
    return new_worker_list

async def error_check_response(writer : StreamWriter, response : dict):
    if 'FORCE_KILL' in response:
        print(f"ERROR IN WORKER {writer.get_extra_info('peername')}") 
        print(f"Closing connection") 
        writer.close()


async def get_worker_mem(reader : StreamReader, writer : StreamWriter):
    request = {'GET_CUDA_MEM': True}
    await Server.send_data(writer, request)
    response = await Server.recieve_data(reader)
    return response

async def INIT_DISTRIB(workers : list):
    print("Getting Worker Mem Stats.")
    worker_mem = await asyncio.gather(*[get_worker_mem(worker.reader, worker.writer) for worker in workers])
    print(f"Mem Stats: {worker_mem}")
    assert len(worker_mem) == len(workers) , "Must receive memory for all workers"
    for i in range(len(workers)):
        await error_check_response(workers[i].writer, worker_mem[i])
        workers[i].CUDA_MEM = worker_mem[i]['CUDA_MEM']

    

async def main():
    assert len(sys.argv) == 2, "USAGE: bossman.py num_workers"
    num_workers = int(sys.argv[1])
    workers = await setup_host(num_workers)
    await INIT_DISTRIB(workers)
    for worker in workers:
        await Server.send_data(worker.writer, CLOSE_CONNECTION)
        worker.writer.close()
        await worker.writer.wait_closed()



if __name__ == "__main__":
    asyncio.run(main())
    
