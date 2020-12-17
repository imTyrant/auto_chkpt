from os import stat
from typing import Dict
from . import functional as F

class RandStateSnapshot(object):
    @staticmethod
    def take():
        state = {
            "numpy": F.get_numpy_rand_state(),
            "torch": F.get_torch_rand_state(),
            "cuda": F.get_cuda_rand_state()
        }
        return state

    @staticmethod
    def set(state:Dict):
        if "numpy" in state:
            F.set_numpy_rand_state(state["numpy"])
        
        if "torch" in state:
            F.set_torch_rand_state(state["torch"])
        
        if "cuda" in state:
            F.set_cuda_rand_state(state["cuda"])