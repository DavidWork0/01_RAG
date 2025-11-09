#Get HW specifications and decide which model version to load
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
import torchvision.transforms as T
import psutil
import GPUtil
import os

def get_hw_config():
    # Get CPU info
    cpu_count = psutil.cpu_count(logical=False)
    total_ram = psutil.virtual_memory().total / (1024 ** 3)  # in GB

    # Get GPU info
    gpus = GPUtil.getGPUs()
    gpu_info = []
    for gpu in gpus:
        gpu_info.append({
            'id': gpu.id,
            'name': gpu.name,
            'total_memory': gpu.memoryTotal,  # in MB
            'free_memory': gpu.memoryFree,    # in MB
            'used_memory': gpu.memoryUsed      # in MB
        })

    hw_config = {
        'cpu_count': cpu_count,
        'total_ram_gb': total_ram,
        'gpus': gpu_info
    }

    return hw_config