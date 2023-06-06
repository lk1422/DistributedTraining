import sys
import os
from typing import List, Tuple
from torch import nn
from asyncio import StreamReader, StreamWriter
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Auxilary
import Server

CLOSE_CONNECTION = {'KILL': True}

class Worker:
    def __init__(self, address : tuple , reader : StreamReader, 
                 writer :StreamWriter ):
        self.address = tuple
        self.reader = reader
        self.writer = writer
        self.CUDA_MEM = None

class BossMan():
    def __init__(self, workers: List[Worker], network : List[Tuple[nn.Module, dict, Tuple]]):
        self.workers = workers
        self.num_workers = len(workers)
        self.network = network
        self.network_stats = []
        self.network_size = 0
    
    async def setup_host(num_connections : int):
        await Server.get_connections(num_connections)
        workers = Server.OPEN_SOCKETS

        new_worker_list = []
        for worker in workers:
            worker_obj = Worker(worker[0], worker[1], worker[2])
            new_worker_list.append(worker_obj)
        return new_worker_list

    async def get_worker_mem(reader : StreamReader, writer : StreamWriter):
        request = {'GET_CUDA_MEM': True}
        await Server.send_data(writer, request)
        response = await Server.recieve_data(reader)
        return response

    async def get_worker_mem_stats(self):
        print("Getting Worker Mem Stats.")
        worker_mem = await asyncio.gather(*[BossMan.get_worker_mem(worker.reader, worker.writer) 
                                            for worker in self.workers])
        print(f"Mem Stats: {worker_mem}")
        assert len(worker_mem) == self.num_workers , "Must receive memory for all workers"
        for i in range(self.num_workers):
            await error_check_response(self.workers[i].writer, worker_mem[i])
            self.workers[i].CUDA_MEM = worker_mem[i]['CUDA_MEM']
    
    def get_model_stats(self):
        model_total_memory = 0
        for module,kwargs, input_shape in self.network:
            init_module = module(**kwargs)
            stat = Auxilary.get_module_memory(init_module, input_shape)
            model_total_memory+=stat
            self.network_stats.append(stat)
        self.network_size = model_total_memory

    async def distribute_model(self):
        """Distribute model amoung workers"""
        get_mem_stats = asyncio.create_task(self.get_worker_mem_stats())
        self.get_model_stats()
        await get_mem_stats
        print(f"total model size: {self.network_size}")
        print("calculating model split")
        target_distribution = Auxilary.calculate_target_distribution(self.workers, self.network_size)
        print(f"model split: {target_distribution}")
        chunks = Auxilary.create_chunks(self.network, self.network_stats, target_distribution,
                                        self.network_size, self.workers)
        print("Calculated Chunks:")
        print(chunks)
        print("Sending Models to Workers")
        chunks = [{'MODEL_DATA': chunk} for chunk in chunks]
        responses = await asyncio.tasks.gather(
            *[(Server.send_recieve(worker.reader, worker.writer, chunk)) 
             for worker,chunk in zip(self.workers, chunks)]
             )
        for worker, response in zip(self.workers, responses):
            await error_check_response(worker.writer, response)
            print(response)
    
    async def model_train(self):
        data = {"TRAIN": True}
        await asyncio.gather(*[Server.send_data(worker.writer,data) 
                               for worker in self.workers ])

        

async def error_check_response(writer : StreamWriter, response : dict):
    if 'FORCE_KILL' in response:
        print(f"ERROR IN WORKER {writer.get_extra_info('peername')}") 
        print(f"Closing connection") 
        writer.close()



async def main():
    assert len(sys.argv) == 2, "USAGE: bossman.py num_workers"
    num_workers = int(sys.argv[1])
    workers = await BossMan.setup_host(num_workers)
    #Test Distribution
    test_model = [
        (nn.Linear, {'in_features':100, 'out_features':512}, (32,100)),
        (nn.Linear, {'in_features':512, 'out_features':1024}, (32, 512)),
        (nn.Linear, {'in_features':1024, 'out_features':1024}, (32,1024)),
        (nn.Linear, {'in_features':1024, 'out_features':1024}, (32, 1024)),
        (nn.Linear, {'in_features':1024, 'out_features':1024}, (32, 1024)),
        (nn.Linear, {'in_features':1024, 'out_features':1024}, (32, 1024)),
        (nn.Linear, {'in_features':1024, 'out_features':1024}, (32, 1024)),
        (nn.Linear, {'in_features':1024, 'out_features':1}, (32, 1024))
        ]
    boss = BossMan(workers, test_model)
    await boss.distribute_model()
    await boss.model_train()

    for worker in workers:
        await Server.send_data(worker.writer, CLOSE_CONNECTION)
        worker.writer.close()
        await worker.writer.wait_closed()




if __name__ == "__main__":
    asyncio.run(main())
    
