from typing import Optional, Union, List
import torch
import numpy as np
import copy


"""
    Here are stuffs for supporting deterministic experiments. Including getting 
    and setting random states from/for 'numpy', 'torch', 'torch.cuda', etc.
"""
## GETTING RNG STATE ##
def get_numpy_rand_state():
    return copy.deepcopy(np.random.get_state())

def get_torch_rand_state()->np.ndarray:
    bt = copy.deepcopy(torch.random.get_rng_state()) #type:torch.ByteTensor
    return bt.numpy()

def get_cuda_rand_state(device:Optional[Union[torch.device, str]]="cuda")->np.ndarray:
    if not torch.cuda.is_available():
        return np.array([])
    if isinstance(device, str):
        device = torch.device(device)
    bt = copy.deepcopy(torch.cuda.get_rng_state(device)) #type:torch.ByteTensor
    return bt.numpy()

def get_all_cuda_rand_state()->List[np.ndarray]:
    if not torch.cuda.is_available():
        return []
    bts = torch.cuda.get_rng_state_all() #type:List[torch.ByteTensor]
    return [copy.deepcopy(bt).numpy() for bt in bts]


## SETTING RNG STATE ##
def set_numpy_rand_state(state):
    np.random.set_state(state)

def set_torch_rand_state(state:np.ndarray):
    torch.random.set_rng_state(torch.from_numpy(state))

def set_cuda_rand_state(state:np.ndarray, device:Optional[Union[torch.device, str]]="cuda"):
    if torch.cuda.is_available():
        if isinstance(device, str):
            device = torch.device(device)
        torch.cuda.set_rng_state(torch.from_numpy(state), torch.device(device))

# Use with caution. Manually setting states for all GPUs may mess up others' experiments.
def set_all_cuda_rand_state(states:List[np.ndarray]):
    if not torch.cuda.is_available():
        return
    if len(states) != torch.cuda.device_count():
        raise ValueError("The number of given states should be equal to the number of GPUs")
    torch.cuda.set_rng_state_all([torch.from_numpy(st) for st in states])
    