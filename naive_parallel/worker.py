import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Server
import asyncio
import Auxilary
import traceback
from typing import List, Tuple
from torch import nn, optim, Tensor
from asyncio import StreamReader, StreamWriter


HOST = os.environ['MODEL_HOST']
PORT = os.environ['MODEL_PORT']

class WorkerNode():
    def __init__(self, reader : StreamReader, writer : StreamWriter):
        self.reader = reader
        self.writer = writer
        self.MODEL = None              #List containing all nn.Modules in order of calling
        self.OPTIMIZER = None          #Optimizer used for this Model
        self.TEMP_FORWARD_MEM = {}     #Store Extra Variables which other modules/nodes depend on
        self.TEMP_BACKWARD_MEM = {}    #Store The gradients of the inputs for the backward pass
        self.EXTERNAL_DEPENDCIES = {}  #Store a list of variables which must be sent out to remote devices
        self.MAX_ERRORS = 1

    def THROW_ERR(error_message : str):
        return {"ERROR": error_message}

    def construct_model(model_received : List[Tuple[nn.Module, dict]]):
        model = []
        for module,kwargs,_ in model_received:
            model.append(module(**kwargs))
        return nn.ModuleList(model)

    def construct_optimizer(self, optimizer_recieved : Tuple[optim.Optimizer, dict]):
        assert self.MODEL!=None, "Model must be sent first"
        optim,kwargs = optimizer_recieved
        params = []
        for module in self.MODEL:
            params.append(module.parameters())
        kwargs['params'] = params
        return optim(**kwargs)

    def load_local_mem(self, incoming_varaibles : dict):
        for (key,val) in incoming_varaibles.items():
            self.TEMP_FORWARD_MEM[key] = val

    def load_external_mem(self, outgoing_names : List[str]):
        for name in outgoing_names:
            self.EXTERNAL_DEPENDCIES[name] = None

    def construct_incoming_input(self, module : nn.Module):
        if hasattr(module, 'forward_order'):
            return [self.TEMP_FORWARD_MEM[input_name] for input_name in module.forward_order]
        return []

    def insert_into_memory(self, variables : List[Tuple[str, Tensor]]):
        for var_name, value in variables:
            if var_name in self.EXTERNAL_DEPENDCIES:
                self.EXTERNAL_DEPENDCIES[var_name] = value
            else:
                self.TEMP_FORWARD_MEM[var_name] = value

    def clear_memory(self):
        self.TEMP_FORWARD_MEM.clear()
        self.TEMP_BACKWARD_MEM.clear()
        self.EXTERNAL_DEPENDCIES.clear()

    def forward(self, model_input : dict):
        x = model_input['input'] #direct input
        self.load_local_mem(model_input['incoming'])
        self.load_external_mem(model_input['outgoing'])
        for module_num, module in enumerate(self.MODEL):
            if (hasattr(module, "out_variables") and len(module.out_variables)):
                x, outgoing = module(x, *self.construct_incoming_input(module))
            else:
                x = module(x, *self.construct_incoming_input(module))
                outgoing = []
            self.insert_into_memory(outgoing)
            self.TEMP_BACKWARD_MEM[module_num] = x
        return {"MODEL_OUT": x, "EXTERNAL_VARIABLES" : self.EXTERNAL_DEPENDCIES}



    def handle_server_request(self, data: dict):
        """
        Returns the data for a corresponding request from the server
        this returned data should be sent back to the server
        """
        print(f"Request Recieved: {data}")

        if 'GET_CUDA_MEM' in data:
            return {'CUDA_MEM': Auxilary.get_device_memory()}
        
        elif 'MODEL_DATA' in data:
            self.MODEL = WorkerNode.construct_model(data['MODEL_DATA'])
            print(self.MODEL)
            return {'MODEL_LOADED': True}
        
        elif 'OPTIMIZER' in data:
            self.OPTIMIZER = self.construct_optimizer(data)
            return {'OPTIMIZER_LOADED': True}

        elif 'FORWARD' in data:
            if self.MODEL == None:
                return WorkerNode.THROW_ERR("model not recieved before forward call")
            return self.forward(data['FORWARD'])

        elif 'BACKWARD' in data:
            raise NotImplementedError

        elif 'EVAL' in data:
            for section in self.MODEL:
                section.eval()

        elif 'TRAIN' in data:
            for section in self.MODEL:
                section.train()

        elif 'SEND_MODEL' in data:
            return self.MODEL

        elif 'KILL' in data:
            return 'KILL'

        else:
            return self.THROW_ERR("REQUEST NOT RECOGNIZED")

    async def client_loop(self):
        ERR_COUNT = 0
        while(True):
            data = await Server.recieve_data(self.reader)
            try:
                response = self.handle_server_request(data)

                if response == 'KILL':
                    print('Shutting Down worker')
                    self.writer.close()
                    await self.writer.wait_closed()
                    break
                elif response is not None:
                    await Server.send_data(self.writer, response)

            except:
                if (ERR_COUNT >= self.MAX_ERRORS):
                    info = traceback.format_exc()
                    await Server.send_data(self.writer, self.THROW_ERR(info))
                    ERR_COUNT+=1
                else:
                    await Server.send_data(self.writer, {"FORCE_KILL": "MAX ERRORS HIT"})
                    self.writer.close()
                    await self.writer.wait_closed()
                    raise

async def main():
    print("Connecting to bossman")
    reader,writer = await asyncio.open_connection(HOST,PORT)
    print("connected to bossman")
    worker = WorkerNode(reader,writer)

    await worker.client_loop()

if __name__ == "__main__":
    asyncio.run(main())

