import os
import sys
import asyncio
from torch import nn
import torch
from typing import List, Tuple
from torch import nn, optim, Tensor
from asyncio import StreamReader, StreamWriter

import Auxilary
import Server

CLOSE_CONNECTION = {'KILL': True}


class Worker:
    """
    Wraps all relavent information about a specific worker for the server
    """
    def __init__(self, address : tuple , reader : StreamReader, 
                 writer :StreamWriter ):
        self.address = address
        self.reader = reader
        self.writer = writer
        self.model_chunk = None
        self.CUDA_MEM = None
        self.incoming_variables = []
        self.outgoing_variables = []

class BossMan():
    def __init__(self, workers: List[Worker], network : List[Tuple[nn.Module, dict, Tuple]]):
        """
        The network is stored as a list containing each module, parameters, and input size
        for the case when the whole network will not fit in memory and must be loaded in pieces
        network = [(module, kwargs, input_size) ...]
        """
        self.workers = workers
        self.num_workers = len(workers)
        self.network = network
        self.module_mem_sizes = []
        self.network_size = 0
        self.variable_to_node = {}
        self.memory = {}

    async def setup_host(num_connections : int) -> List[Worker]:
        """
            Waits until the server has connected with num_connections workers. Then returns
            a list storing Worker objects specifying to each connection.
        """
        await Server.get_connections(num_connections)
        workers = Server.OPEN_SOCKETS

        new_worker_list = []
        for worker in workers:
            worker_obj = Worker(worker[0], worker[1], worker[2])
            new_worker_list.append(worker_obj)
        return new_worker_list

    async def get_worker_mem(reader : StreamReader, writer : StreamWriter) -> dict:
        request = {'GET_CUDA_MEM': True}
        await Server.send_data(writer, request)
        response = await Server.recieve_data(reader)
        return response

    async def get_worker_mem_stats(self) -> None:
        worker_mem = await asyncio.gather(*[BossMan.get_worker_mem(worker.reader, worker.writer) 
                                            for worker in self.workers])
        assert len(worker_mem) == self.num_workers , "Must receive memory for all workers"
        for i in range(self.num_workers):
            await error_check_response(self.workers[i].writer, worker_mem[i])
            self.workers[i].CUDA_MEM = worker_mem[i]['CUDA_MEM']
    
    def get_model_stats(self) -> None:
        """
        Collects the total amount of memory each module takes to store
        during training.
        """
        model_total_memory = 0
        for module,kwargs, input_shape in self.network:
            init_module = module(**kwargs)
            module_mem_size = Auxilary.get_module_memory(init_module, input_shape)
            model_total_memory+=module_mem_size
            self.module_mem_sizes.append(module_mem_size)
        self.network_size = model_total_memory

    async def distribute_model(self) -> None:
        """Distribute model amoung workers"""
        get_mem_stats = asyncio.create_task(self.get_worker_mem_stats())
        self.get_model_stats()
        await get_mem_stats
        target_distribution = Auxilary.calculate_target_distribution(self.workers)
        chunks = Auxilary.create_chunks(self.network, self.module_mem_sizes, target_distribution,
                                        self.network_size, self.workers)
        chunks = [{'MODEL_DATA': chunk} for chunk in chunks]
        responses = await asyncio.tasks.gather(
            *[(Server.send_recieve(worker.reader, worker.writer, chunk)) 
             for worker,chunk in zip(self.workers, chunks)]
             )
        for i in range(len(self.workers)):
            await error_check_response(self.workers[i].writer, responses[i])
            self.workers[i].model_chunk = chunks[i]["MODEL_DATA"]

    def gather_dependencies(self) -> None:
        """
        Stores the incoming and outgoing variables from a specific worker node.
        """
        for worker in self.workers:
            for layer in worker.model_chunk:
                layer_init = layer[0](**layer[1])
                if hasattr(layer_init, "dependencies"):
                    worker.incoming_variables.extend(layer_init.dependencies)
                if hasattr(layer_init, "out_variables"):
                    worker.outgoing_variables.extend(layer_init.out_variables)
            for incoming in worker.incoming_variables:
                if incoming in worker.outgoing_variables:
                    worker.incoming_variables.remove(incoming)
                    worker.outgoing_variables.remove(incoming)

    def load_memory(self, data : List[Tuple[str, Tensor]]) -> None:
        """
        Loads a list of variables into the bossmans memory
        """
        for var_name,value in data:
            self.memory[var_name] = value

    def load_memory_grads(self, data : List[Tuple[str, Tensor]]) -> None:
        """
        Loads a list of variable's gradients into their respecitve tensors
        stored in bossmans memory.
        """
        for name, value in data:
            self.memory[name].grad = value
    
    
    async def forward(self, x : Tensor, extra_inputs : dict) -> Tensor:
        """
        Preforms the forward pass on the model.

        Parameters:
            extra_inputs - a dictionary storing extra input variables for the model.
        """
        self.load_memory(extra_inputs)
        for worker in self.workers:

            outgoing = worker.outgoing_variables
            incoming = {name:self.memory[name] for name in worker.incoming_variables}
            request = { 
                "FORWARD": {
                    "input":x,
                    "incoming":incoming,
                    "outgoing":outgoing
                }
            }
            response = await Server.send_recieve(worker.reader, worker.writer, request)
            await error_check_response(worker.writer, response)
            x = response["MODEL_OUT"]
            self.load_memory(response["EXTERNAL_VARIABLES"])
        return x
    
    async def backward(self, out_grad : Tensor):
        for worker in reversed(self.workers):
            incoming = [(name,self.memory[name].grad) for name in worker.outgoing_variables]
            request = { 
                "BACKWARD": {
                    "NODE_OUT":out_grad,
                    "GRADS":incoming
                }
            }
            response = await Server.send_recieve(worker.reader, worker.writer, request)
            await error_check_response(worker.writer, response)
            out_grad = response["NODE_IN"]
            self.load_memory_grads(response["GRADS"])
    
    async def send_optimizer(self, optim, kwargs):
        request = {"OPTIMIZER": (optim, kwargs)}
        responses = await asyncio.gather(*[Server.send_recieve(worker.reader, worker.writer,request) 
                               for worker in self.workers ])
        for worker, response in zip(self.workers, responses):
            await error_check_response(worker, response)

    async def model_train(self):
        request = {"TRAIN": True}
        responses = await asyncio.gather(*[Server.send_recieve(worker.reader, worker.writer,request) 
                               for worker in self.workers ])
        for worker, response in zip(self.workers, responses):
            await error_check_response(worker, response)

    async def optim_step(self):
        request = {"OPTIM_STEP": True}
        responses = await asyncio.gather(*[Server.send_recieve(worker.reader, worker.writer,request) 
                               for worker in self.workers ])
        for worker, response in zip(self.workers, responses):
            await error_check_response(worker, response)

    async def model_eval(self):
        request = {"EVAL": True}
        responses = await asyncio.gather(*[Server.send_recieve(worker.reader, worker.writer,request) 
                               for worker in self.workers ])
        for worker, response in zip(self.workers, responses):
            await error_check_response(worker, response)

    async def node_mem_clear(self):
        request = {"CLEAR_MEMORY": True}
        responses = await asyncio.gather(*[Server.send_recieve(worker.reader, worker.writer,request) 
                               for worker in self.workers ])
        for worker, response in zip(self.workers, responses):
            await error_check_response(worker, response)

async def error_check_response(writer : StreamWriter, response : dict):
    if 'FORCE_KILL' in response:
        print(f"ERROR IN WORKER {writer.get_extra_info('peername')}") 
        print(f"Closing connection") 
        writer.close()
        raise Exception
