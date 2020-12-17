from typing import Union
import torch
import os
import copy
import numpy as np
import pickle
import logging

from torch.nn import Module
from torch.optim.optimizer import Optimizer
from auto_chkpt.base_saver import BaseSaver
from auto_chkpt.utils.exception import AutoCheckpointSaverException
from auto_chkpt.rand_state import RandStateSnapshot

def _symlink(src, dst, overwrite=True):
    if not overwrite:
        os.symlink(os.path.abspath(src), os.path.abspath(dst))
        return
    
    if os.path.exists(dst) or os.path.islink(dst):
        os.remove(dst)
    os.symlink(os.path.abspath(src), os.path.abspath(dst))
    pass

def _exists(path):
    return os.path.exists(path) or os.path.islink(path)

def _real_path(path):
    if os.path.islink(path):
        return os.readlink(path)
    else:
        return path


class SimpleSaver(BaseSaver):
    MODEL_KEY = "model"
    OPTIMIZER_KEY = "optimizer"
    EPOCH_KEY = "epoch"

    SNAPSHOT_EXT = "mem"
    CHECKPOINT_EXT = "ckpt"
    RNG_STATE_EXT = "rand"
    EXCEPTION_EXT = "exce"

    def __init__(self,
            chkpt_fold:str,
            tag:str,
            model:Module,
            optimizer:Optimizer = None,
            device:Union[str,torch.device]="cuda",
            chkpt_steps:int=1,
            start_step:int=0,
            resume:bool=True,
            single_chkpt_file:bool=False,
            symbolic_link:bool=True,
            save_rng_state:bool=True,
            memory_snapshot:bool=True,
        ) -> None:

        super(SimpleSaver, self).__init__()

        # Assuming no parallelism is used.
        if isinstance(device, str): 
            device = torch.device(device)
        self._device = device

        # Naming checkpoint stuffs
        self._tag = tag

        # Directory for storing checkpoint stuffs
        self._chkpt_fold = chkpt_fold
        if not os.path.exists(self._chkpt_fold):
            os.mkdir(self._chkpt_fold)

        # Model
        self._model = model

        # Optimizer
        self._optimizer = optimizer

        # By 'chkpt_steps' steps, saver automatically make a chkpt
        self._chkpt_steps = chkpt_steps

        # The epoch from which the saver starts (default is 0).
        self._start_step = start_step

        # Lastest epoch, for making chkpt.
        self._cur_step = self._start_step

        # Saving each chkpt file seperately.
        self._single_chkpt_file = single_chkpt_file

        # Create a symbolic link pointing to the latest checkpoint.
        # (If 'single_chkpt_file' is True, this one can be ignored).
        self._symbolic_link = symbolic_link
        if self._single_chkpt_file:
            self._symbolic_link = False

        # Saving rng state (including 'torch', 'cuda' and 'numpy').
        self._save_rand_state = save_rng_state

        # Snapshot for rng state. (If 'save_rng_state' is False, it can be ignored).
        self._rand_state = None

        # Saving snapshot in memory. (If given False, saver will store snapshots in disk for saving memory.)
        self._memory_snapshot = memory_snapshot

        # Snapshot for states of model, optimizer and epoch.
        # NOTE: Here is an ugly hack that '_training_state' is a 'Dict' or a 'str'.
        if self._memory_snapshot:
            self._training_state = None
        else:
            self._training_state = self._make_path(SimpleSaver.SNAPSHOT_EXT)

        self._logger = logging.getLogger()
        self._logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        self._logger.addHandler(ch)

        if resume:
            self._logger.info("Try to load existing checkpoint..")
            if _exists(self._make_path(SimpleSaver.CHECKPOINT_EXT)) and \
                (not self._save_rand_state or _exists(self._make_path(SimpleSaver.RNG_STATE_EXT))):
                self.load()
            self._logger.info(f"Start process from {self.epoch}")

    def set_logger(self, logger):
        self._logger = logger
    
    @property
    def epoch(self):
        return self._cur_step

    def step(self):
        self._cur_step += 1
        self._take_mem_snapshot()
        interval = self._cur_step - self._start_step
        if interval != 0 and interval % self._chkpt_steps == 0:
            self.save()
    
    def save(self) -> None:
        try:
            if not self._single_chkpt_file:
                self._saving_training_state_proc(self._make_path(f"{SimpleSaver.CHECKPOINT_EXT}.{self.epoch}"))
                if self._save_rand_state:
                    self._saving_rand_state_proc(self._make_path(f"{SimpleSaver.RNG_STATE_EXT}.{self.epoch}"))

            if self._symbolic_link:
                _symlink(self._make_path(f"{SimpleSaver.CHECKPOINT_EXT}.{self.epoch}"), self._make_path(SimpleSaver.CHECKPOINT_EXT))
                if self._save_rand_state:
                    _symlink(self._make_path(f"{SimpleSaver.RNG_STATE_EXT}.{self.epoch}"), self._make_path(SimpleSaver.RNG_STATE_EXT))
            else:
                self._saving_training_state_proc(self._make_path(SimpleSaver.CHECKPOINT_EXT))
                if self._save_rand_state:
                    self._saving_rand_state_proc(self._make_path(SimpleSaver.RNG_STATE_EXT))

            self._logger.info(f"Checkpointing, current epoch {self.epoch}")
        except Exception as e:
            raise AutoCheckpointSaverException("'save' failed")
    
    def exception(self):
        try:
            self._logger.info(f"Catch exception, {self._cur_step}")
            self._saving_training_state_proc(self._make_path(f"{SimpleSaver.CHECKPOINT_EXT}.{SimpleSaver.EXCEPTION_EXT}"))
            if self._save_rand_state:
                self._saving_rand_state_proc(self._make_path(f"{SimpleSaver.RNG_STATE_EXT}.{SimpleSaver.EXCEPTION_EXT}"))

        except Exception as e:
            raise AutoCheckpointSaverException("'exception' failed")
    
    def load(self):
        try:
            training_state = _real_path(self._make_path(SimpleSaver.CHECKPOINT_EXT))
            self._load_training_state(training_state)
            self._logger.info(f"Load and set training state from {training_state}")
            if self._save_rand_state:
                rand_state = _real_path(self._make_path(SimpleSaver.RNG_STATE_EXT))
                self._load_rand_state(rand_state)
                self._logger.info(f"Load and set rand state from {rand_state}")
        except Exception as e:
            raise AutoCheckpointSaverException("'load' failed")

    ## private methods ##
    # make file path for checkpoint
    def _make_path(self, ext):
        return os.path.abspath(os.path.join(self._chkpt_fold, f"{self._tag}.{ext}"))
        
    # take snapshot and save in memory
    def _take_mem_snapshot(self):
        if self._memory_snapshot:
            self._training_state = {
                SimpleSaver.MODEL_KEY: copy.deepcopy(self._model.state_dict()),
                SimpleSaver.OPTIMIZER_KEY: None if self._optimizer is None else copy.deepcopy(self._optimizer.state_dict()),
                SimpleSaver.EPOCH_KEY: self.epoch
            }
        else:
            torch.save({
                SimpleSaver.MODEL_KEY: self._model.state_dict(),
                SimpleSaver.OPTIMIZER_KEY: None if self._optimizer is None else self._optimizer.state_dict(),
                SimpleSaver.EPOCH_KEY: self.epoch
            }, self._training_state)
        
        if self._save_rand_state:
            self._rand_state = RandStateSnapshot.take()
    
    # saving
    def _saving_training_state_proc(self, path):
        if self._memory_snapshot:
            torch.save(self._training_state, path)
        else:
            import shutil
            shutil.copy2(self._training_state, path)
    
    def _saving_rand_state_proc(self, path):
        with open(path, "wb") as f:
            pickle.dump(self._rand_state, f)
    
    # loading
    def _load_training_state(self, path):
        if self._memory_snapshot:
            self._training_state = torch.load(path, map_location=self._device)
            obj = self._training_state
        else:
            import shutil
            shutil.copy2(path, self._training_state)
            obj = torch.load(self._training_state, map_location=self._device)
        
        self._cur_step = obj[SimpleSaver.EPOCH_KEY]
        self._model.load_state_dict(obj[SimpleSaver.MODEL_KEY])
        if self._optimizer is not None:
            self._optimizer.load_state_dict(obj[SimpleSaver.OPTIMIZER_KEY])

    
    def _load_rand_state(self, path):
        with open(path, "rb") as f:
            self._rand_state = pickle.load(f)
        RandStateSnapshot.set(self._rand_state)