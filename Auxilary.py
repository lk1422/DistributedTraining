import torch
import torch.nn as nn
from torchinfo import summary
from typing import Tuple, List


def get_module_memory(module : nn.Module, input_size : Tuple) -> int:
    stats = summary(module, input_size=input_size, verbose=0)

    return (
        stats.total_input + stats.total_output_bytes + stats.total_param_bytes
    )

def get_device_memory():
    """
    Returns Total GPU Memory in bytes
    """
    assert torch.cuda.is_available(), "CUDA NOT AVAILABLE FOR THIS DEVICE" 
    device = torch.device('cuda')
    return torch.cuda.get_device_properties(device).total_memory


def calculate_target_distribution(workers : List , network_memsize : int) -> List[float]:
    memories = [worker.CUDA_MEM for worker in workers]
    total_available_memory = sum(memories)
    print(f"Model takes up {network_memsize/total_available_memory} of memory capacity")
    assert total_available_memory > network_memsize , "Not Enough Memory"
    memory_distribution = []
    for device_mem in memories:
        device_network_mem = (device_mem/total_available_memory)
        memory_distribution.append(device_network_mem)
    return memory_distribution

def create_chunks(model: List[Tuple[nn.Module, dict]], model_stats : List[int], 
                  target_distribution: List[float], network_size:int,
                  workers: List):
    module = 0
    device = 0
    chunks = []
    while device < len(workers) and module < len(model):
        current_chunk = []
        module_memory = 0
        target_mem = network_size*target_distribution[device]
        while module_memory < target_mem and module != len(model):
            new_module_mem = model_stats[module] + module_memory
            if (new_module_mem < workers[device].CUDA_MEM):
                current_chunk.append(model[module])
                module_memory = new_module_mem
                module+=1
            else:
                break
        chunks.append(current_chunk)
        device+=1
    
    assert module>=len(model), "nn.Module's in list are to large to break up optimally"
    return chunks


            

