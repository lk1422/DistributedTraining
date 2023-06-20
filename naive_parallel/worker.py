import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Server
import asyncio
import Auxilary
import traceback
from typing import List, Tuple
from torch import nn, optim, Tensor
import torch
from asyncio import StreamReader, StreamWriter


HOST = os.environ['MODEL_HOST']
PORT = os.environ['MODEL_PORT']
GPU = torch.device('cuda')
CPU = torch.device('cpu')

class WorkerNode():
    def __init__(self, reader : StreamReader, writer : StreamWriter):
        self.reader = reader
        self.writer = writer
        self.MODEL = None              #List containing all nn.Modules in order of calling
        self.OPTIMIZER = None          #Optimizer used for this Model
        self.TEMP_LOCAL_MEM = {}       #Store Extra Variables which other modules/nodes depend on
        self.incoming_variables = []   #List containing all incoming variable names
        self.EXTERNAL_DEPENDCIES = {}  #Store a list of variables which must be sent out to remote devices
        self.MAX_ERRORS = 1

    def THROW_ERR(error_message : str):
        return {"ERROR": error_message}

    def construct_model(model_received : List[Tuple[nn.Module, dict]]):
        model = []
        for module,kwargs,_ in model_received:
            model.append(module(**kwargs).to(GPU))
        return nn.ModuleList(model)

    def construct_optimizer(self, optimizer_recieved : Tuple[optim.Optimizer, dict]):
        assert self.MODEL!=None, "Model must be sent first"
        optim,kwargs = optimizer_recieved
        params = []
        for module in self.MODEL:
            params.append(module.parameters())
        kwargs['params'] = params
        return optim(**kwargs)

    def load_local_mem(self, incoming_variables : dict):
        for (key,val) in incoming_variables.items():
            self.TEMP_LOCAL_MEM[key] = val.to(GPU)
        
    def load_incoming_varibles(self, incoming_variables : dict):
        self.load_local_mem(incoming_variables)
        self.incoming_variables = [key for key in incoming_variables.keys()]


    def load_external_mem(self, outgoing_names : List[str]):
        for name in outgoing_names:
            self.EXTERNAL_DEPENDCIES[name] = None

    def construct_incoming_input(self, module : nn.Module):
        if hasattr(module, 'forward_order'):
            return [self.get_from_memory(input_name) for input_name in module.forward_order]
        return []

    def insert_into_memory(self, variables : List[Tuple[str, Tensor]]):
        for var_name, value in variables:
            if var_name in self.EXTERNAL_DEPENDCIES:
                self.EXTERNAL_DEPENDCIES[var_name] = value
            else:
                self.TEMP_LOCAL_MEM[var_name] = value

    def clear_memory(self):
        self.TEMP_LOCAL_MEM.clear()
        self.EXTERNAL_DEPENDCIES.clear()

    def clear_dependencies(self):
        self.EXTERNAL_DEPENDCIES.clear()
    
    def get_from_memory(self, variable_name : str):
        if variable_name in self.TEMP_LOCAL_MEM:
            return self.TEMP_LOCAL_MEM[variable_name]
        elif variable_name in self.EXTERNAL_DEPENDCIES:
            return self.EXTERNAL_DEPENDCIES[variable_name]
        return None
            
    def insert_grads_into_memory(self, out_gradients: List[Tuple[str, Tensor]]):
       for name,value in out_gradients:
           if value != None:
               self.EXTERNAL_DEPENDCIES[name].grad = value.to(GPU)

    
    def mem_to_cpu(self, memory):
        return [(key,value.to(CPU)) for key,value in memory.items()]
    
    def grad_to_cpu(self, grad):
        if grad == None:
            return grad
        return grad.to(CPU)
    
    def retain_incoming_variables(self):
        for name in self.incoming_variables:
            if self.get_from_memory(name).requires_grad:
                self.get_from_memory(name).retain_grad()
    
    def forward(self, model_input : dict):
        x = model_input['input'].to(GPU) #direct input
        self.load_incoming_varibles(model_input['incoming'])
        self.load_external_mem(model_input['outgoing'])
        for module_num, module in enumerate(self.MODEL):
            if(x.requires_grad):
                x.retain_grad()
            self.TEMP_LOCAL_MEM[str(module_num) + "in"] = x
            if (hasattr(module, "out_variables") and len(module.out_variables)):
                x, outgoing = module(x, *self.construct_incoming_input(module))
            else:
                x = module(x, *self.construct_incoming_input(module))
                outgoing = []
            self.insert_into_memory(outgoing)
            self.TEMP_LOCAL_MEM[str(module_num) + "out"] = x
        return {
                "MODEL_OUT": x.to(CPU), 
                "EXTERNAL_VARIABLES" : self.mem_to_cpu(self.EXTERNAL_DEPENDCIES)
            }
    
    def backward(self, model_gradients : dict):
        out_grad = model_gradients['NODE_OUT'].to(GPU)
        self.insert_grads_into_memory(model_gradients['GRADS'])
        self.retain_incoming_variables()
        #Computing the gradients
        for module_num in reversed(range(len(self.MODEL))):
            module_out = self.TEMP_LOCAL_MEM[str(module_num)+"out"]
            module_out.grad = out_grad
            module_out.backward(module_out.grad, retain_graph=True)
            out_grad = self.TEMP_LOCAL_MEM[str(module_num)+"in"].grad
        #Collect Gradients of incoming variables
        outgoing_grads = [(name, self.grad_to_cpu(self.get_from_memory(name).grad)) 
                          for name in self.incoming_variables]
        return {
            "NODE_IN": self.grad_to_cpu(out_grad),
            "GRADS": outgoing_grads
            }

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
            response = self.backward(data["BACKWARD"])
            #self.clear_memory() PUT BACK LATER
            return response
        
        elif 'CLEAR_MEMORY' in data:
            self.clear_memory()
            return {"MEMORY_CLEARED" : True}

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

