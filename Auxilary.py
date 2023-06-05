import torch
import torch.nn as nn
from torchinfo import summary
from typing import Tuple, List, NamedTuple


def get_module_memory(module : nn.Module, input_size : Tuple) -> int:
    stats = summary(module, input_size=input_size)
    input_size = stats.to_megabytes(stats.total_input)

    return stats.to_megabytes(
        stats.total_input + stats.total_output_bytes + stats.total_param_bytes
    )

def get_device_memory():
    """
    Returns Total GPU Memory in bytes
    """
    assert torch.cuda.is_available(), "CUDA NOT AVAILABLE FOR THIS DEVICE" 
    device = torch.device('cuda')
    return torch.cuda.get_device_properties(device).total_memory


"""
def calculateTargetMemDistribution(device_memories : List[float] , network_memsize : float) -> List[float]:
    total_available_memory = sum(device_memories)
    memory_distribution = []
    for device_mem in device_memories:
        device_network_mem = (device_mem/total_available_memory) * network_memsize
        memory_distribution.append(device_network_mem)
    return memory_distribution


def createNextChunk( network : List[NamedTuple], target_memsize : float, cap_memsize=float('inf')) ->None:
    current_memsize = 0
    chunk = nn.ModuleList()

    while (current_memsize < target_memsize or len(network) == 0):
        module = network[0].module(**network[0].kwargs)
        current_memsize += getModuleMemory(module)
        chunk.extend([module])
        network.pop(0)

    return chunk
"""
