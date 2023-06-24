import torch
import torch.nn as nn
from torchinfo import summary
from typing import Tuple, List




def get_module_memory(module : nn.Module, input_size : List[Tuple]) -> int:
    """
    Returns the total memory (in bytes) required to store a module in memory
    """
    stats = summary(module, input_size=input_size, verbose=0)
    return (
        stats.total_input + stats.total_output_bytes + stats.total_param_bytes
    )

def get_device_memory() -> int:
    """
    Returns the total amount of available GPU memory on a device
    """
    assert torch.cuda.is_available(), "CUDA NOT AVAILABLE FOR THIS DEVICE" 
    device = torch.device('cuda')
    return torch.cuda.get_device_properties(device).total_memory


def calculate_target_distribution(workers : List) -> List[float]:
    """
    Returns a precentage distribution which specifies how to split the model.
    The precentage is calculated by (DEVICE_MEMORY)/(TOTAL AVAILABLE MEMORY)
    """
    memories = [worker.CUDA_MEM for worker in workers]
    total_available_memory = sum(memories)
    memory_distribution = []
    for device_mem in memories:
        device_network_mem = (device_mem/total_available_memory)
        memory_distribution.append(device_network_mem)
    return memory_distribution

def create_chunks(model: List[Tuple[nn.Module, dict]], module_mem_sizes : List[int], 
                  target_distribution: List[float], network_size:int,
                  workers: List) -> List[List[nn.Module]]:
    """
    Returns a list which at each element stores a list of modules
    a specific worker is responsible for.

    Parameters:
        model - a representation of the model which doesn't force the whole network to 
                be stored in memory all at once. Each index stores a tuple containing 
                a module and the neccessary kwargs to initialize it.
    """
    chunks = []
    module = 0
    device = 0
    while device < len(workers) and module < len(model):
        current_chunk = []
        module_memory = 0
        target_mem = network_size*target_distribution[device]
        while module_memory < target_mem and module != len(model):
            new_module_mem = module_mem_sizes[module] + module_memory
            if (new_module_mem < workers[device].CUDA_MEM):
                current_chunk.append(model[module])
                module_memory = new_module_mem
                module+=1
            else:
                break
        chunks.append(current_chunk)
        device+=1
    
    return chunks


            

