import torch
import gc


def clear_cache():
  torch.cuda.empty_cache()  # Free unused memory
  gc.collect()  # Run Python garbage collector

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device
