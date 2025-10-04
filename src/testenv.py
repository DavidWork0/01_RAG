from llama_cpp.llama_cpp import load_shared_library
import pathlib
import llama_cpp

# Get the library path from the installed package
lib_path = pathlib.Path(llama_cpp.__file__).parent / 'lib'

try:
    lib = load_shared_library('llama', lib_path)
    gpu_supported = bool(lib.llama_supports_gpu_offload())
    print(f"GPU support: {gpu_supported}")
except Exception as e:
    print(f"Error checking GPU support: {e}")
