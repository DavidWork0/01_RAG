"""
Hardware Information Module
============================
Cross-platform hardware detection for Windows and Linux systems.
Collects CPU, GPU, RAM, and system information for test logging.

Author: Generated for 01_RAG project
Date: November 16, 2025
"""

import platform
import subprocess
import sys
from typing import Dict, Optional, List
import psutil


def get_cpu_info() -> Dict[str, str]:
    """
    Get CPU information (cross-platform).
    
    Returns:
        Dictionary with CPU details
    """
    info = {
        'processor': platform.processor() or 'Unknown',
        'architecture': platform.machine(),
        'physical_cores': str(psutil.cpu_count(logical=False)),
        'logical_cores': str(psutil.cpu_count(logical=True)),
        'max_frequency': 'Unknown',
        'current_frequency': 'Unknown'
    }
    
    try:
        cpu_freq = psutil.cpu_freq()
        if cpu_freq:
            info['max_frequency'] = f"{cpu_freq.max:.2f} MHz"
            info['current_frequency'] = f"{cpu_freq.current:.2f} MHz"
    except Exception:
        pass
    
    # Try to get more detailed CPU name
    if platform.system() == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            info['processor'] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
            winreg.CloseKey(key)
        except Exception:
            pass
    elif platform.system() == "Linux":
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if 'model name' in line:
                        info['processor'] = line.split(':')[1].strip()
                        break
        except Exception:
            pass
    
    return info


def get_ram_info() -> Dict[str, str]:
    """
    Get RAM information (cross-platform).
    
    Returns:
        Dictionary with RAM details
    """
    mem = psutil.virtual_memory()
    
    info = {
        'total': f"{mem.total / (1024**3):.2f} GB",
        'available': f"{mem.available / (1024**3):.2f} GB",
        'used': f"{mem.used / (1024**3):.2f} GB",
        'percent_used': f"{mem.percent:.1f}%"
    }
    
    return info


def get_gpu_info() -> List[Dict[str, str]]:
    """
    Get GPU information (cross-platform).
    Supports NVIDIA GPUs via nvidia-smi and attempts AMD/Intel detection.
    
    Returns:
        List of dictionaries with GPU details
    """
    gpus = []
    
    # Try NVIDIA GPU detection
    try:
        if platform.system() == "Windows":
            nvidia_smi = "nvidia-smi"
        else:
            nvidia_smi = "nvidia-smi"
        
        # Query NVIDIA GPUs
        result = subprocess.run(
            [nvidia_smi, '--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 8:
                        gpus.append({
                            'index': parts[0],
                            'name': parts[1],
                            'driver_version': parts[2],
                            'memory_total': f"{parts[3]} MB",
                            'memory_used': f"{parts[4]} MB",
                            'memory_free': f"{parts[5]} MB",
                            'temperature': f"{parts[6]}°C" if parts[6] != 'N/A' else 'N/A',
                            'utilization': f"{parts[7]}%" if parts[7] != 'N/A' else 'N/A',
                            'vendor': 'NVIDIA'
                        })
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    
    # Try PyTorch CUDA detection as fallback
    if not gpus:
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    gpus.append({
                        'index': str(i),
                        'name': props.name,
                        'driver_version': 'N/A',
                        'memory_total': f"{props.total_memory / (1024**2):.0f} MB",
                        'memory_used': 'N/A',
                        'memory_free': 'N/A',
                        'temperature': 'N/A',
                        'utilization': 'N/A',
                        'vendor': 'CUDA-Compatible',
                        'compute_capability': f"{props.major}.{props.minor}"
                    })
        except ImportError:
            pass
    
    # If still no GPU found, return placeholder
    if not gpus:
        gpus.append({
            'index': '0',
            'name': 'No GPU detected or not accessible',
            'driver_version': 'N/A',
            'memory_total': 'N/A',
            'memory_used': 'N/A',
            'memory_free': 'N/A',
            'temperature': 'N/A',
            'utilization': 'N/A',
            'vendor': 'Unknown'
        })
    
    return gpus


def get_disk_info() -> Dict[str, str]:
    """
    Get disk information for the current working directory (cross-platform).
    
    Returns:
        Dictionary with disk details
    """
    import os
    disk = psutil.disk_usage(os.getcwd())
    
    info = {
        'total': f"{disk.total / (1024**3):.2f} GB",
        'used': f"{disk.used / (1024**3):.2f} GB",
        'free': f"{disk.free / (1024**3):.2f} GB",
        'percent_used': f"{disk.percent:.1f}%"
    }
    
    return info


def get_os_info() -> Dict[str, str]:
    """
    Get operating system information (cross-platform).
    
    Returns:
        Dictionary with OS details
    """
    info = {
        'system': platform.system(),
        'release': platform.release(),
        'version': platform.version(),
        'platform': platform.platform(),
        'python_version': sys.version.split()[0],
        'python_implementation': platform.python_implementation()
    }
    
    return info


def get_cuda_info() -> Dict[str, str]:
    """
    Get CUDA information if available.
    
    Returns:
        Dictionary with CUDA details
    """
    info = {
        'available': 'No',
        'version': 'N/A',
        'device_count': '0',
        'current_device': 'N/A'
    }
    
    try:
        import torch
        if torch.cuda.is_available():
            info['available'] = 'Yes'
            info['version'] = torch.version.cuda or 'N/A'
            info['device_count'] = str(torch.cuda.device_count())
            info['current_device'] = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'N/A'
    except ImportError:
        info['available'] = 'PyTorch not installed'
    
    return info


def get_all_hardware_info() -> Dict:
    """
    Get comprehensive hardware information (cross-platform).
    
    Returns:
        Dictionary with all hardware details
    """
    return {
        'os': get_os_info(),
        'cpu': get_cpu_info(),
        'ram': get_ram_info(),
        'gpus': get_gpu_info(),
        'disk': get_disk_info(),
        'cuda': get_cuda_info()
    }


def format_hardware_info(hw_info: Dict) -> str:
    """
    Format hardware information for display.
    
    Args:
        hw_info: Hardware information dictionary
    
    Returns:
        Formatted string
    """
    lines = []
    
    lines.append("HARDWARE INFORMATION")
    lines.append("="*80)
    lines.append("")
    
    # OS Information
    lines.append("Operating System:")
    lines.append("-"*80)
    os_info = hw_info['os']
    lines.append(f"  System:              {os_info['system']}")
    lines.append(f"  Release:             {os_info['release']}")
    lines.append(f"  Version:             {os_info['version']}")
    lines.append(f"  Platform:            {os_info['platform']}")
    lines.append(f"  Python Version:      {os_info['python_version']}")
    lines.append(f"  Python Impl:         {os_info['python_implementation']}")
    lines.append("")
    
    # CPU Information
    lines.append("CPU:")
    lines.append("-"*80)
    cpu_info = hw_info['cpu']
    lines.append(f"  Processor:           {cpu_info['processor']}")
    lines.append(f"  Architecture:        {cpu_info['architecture']}")
    lines.append(f"  Physical Cores:      {cpu_info['physical_cores']}")
    lines.append(f"  Logical Cores:       {cpu_info['logical_cores']}")
    lines.append(f"  Max Frequency:       {cpu_info['max_frequency']}")
    lines.append(f"  Current Frequency:   {cpu_info['current_frequency']}")
    lines.append("")
    
    # RAM Information
    lines.append("RAM:")
    lines.append("-"*80)
    ram_info = hw_info['ram']
    lines.append(f"  Total:               {ram_info['total']}")
    lines.append(f"  Available:           {ram_info['available']}")
    lines.append(f"  Used:                {ram_info['used']}")
    lines.append(f"  Percent Used:        {ram_info['percent_used']}")
    lines.append("")
    
    # GPU Information
    lines.append("GPU(s):")
    lines.append("-"*80)
    for idx, gpu in enumerate(hw_info['gpus'], 1):
        if idx > 1:
            lines.append("")
        lines.append(f"  GPU {gpu['index']}:")
        lines.append(f"    Name:              {gpu['name']}")
        lines.append(f"    Vendor:            {gpu.get('vendor', 'Unknown')}")
        if gpu.get('driver_version') != 'N/A':
            lines.append(f"    Driver Version:    {gpu['driver_version']}")
        lines.append(f"    Memory Total:      {gpu['memory_total']}")
        if gpu['memory_used'] != 'N/A':
            lines.append(f"    Memory Used:       {gpu['memory_used']}")
            lines.append(f"    Memory Free:       {gpu['memory_free']}")
        if gpu.get('temperature') and gpu['temperature'] != 'N/A':
            lines.append(f"    Temperature:       {gpu['temperature']}")
        if gpu.get('utilization') and gpu['utilization'] != 'N/A':
            lines.append(f"    Utilization:       {gpu['utilization']}")
        if gpu.get('compute_capability'):
            lines.append(f"    Compute Cap:       {gpu['compute_capability']}")
    lines.append("")
    
    # CUDA Information
    lines.append("CUDA:")
    lines.append("-"*80)
    cuda_info = hw_info['cuda']
    lines.append(f"  Available:           {cuda_info['available']}")
    lines.append(f"  Version:             {cuda_info['version']}")
    lines.append(f"  Device Count:        {cuda_info['device_count']}")
    if cuda_info['current_device'] != 'N/A':
        lines.append(f"  Current Device:      {cuda_info['current_device']}")
    lines.append("")
    
    # Disk Information
    lines.append("Disk (Current Working Directory):")
    lines.append("-"*80)
    disk_info = hw_info['disk']
    lines.append(f"  Total:               {disk_info['total']}")
    lines.append(f"  Used:                {disk_info['used']}")
    lines.append(f"  Free:                {disk_info['free']}")
    lines.append(f"  Percent Used:        {disk_info['percent_used']}")
    lines.append("")
    
    return "\n".join(lines)


if __name__ == "__main__":
    """Test hardware detection when run directly."""
    print("\nCollecting hardware information...\n")
    hw_info = get_all_hardware_info()
    print(format_hardware_info(hw_info))
