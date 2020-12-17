# from .exception import AutoCheckpointException
import functools
import uuid

from typing import Dict, Set, Union
from auto_chkpt.base_saver import BaseSaver
from auto_chkpt.utils.exception import AutoCheckpointSaverException

def _naming_saver():
    return str(uuid.uuid1())

def _invoke_all_saver(savers:Dict[str, BaseSaver]):
    for each in savers.keys():
        saver = savers[each]
        saver.exception()

def _is_watching_current_exceptions(watching:Set[str], e):
    if "*" in watching:
        return True
    
    if e in watching:
        return True

    return False

class CheckpointRegistry(object):
    catch_all = True
    auto_detach = True
    existing_savers = {} #type:Dict[str, BaseSaver]
    watching_exceptions = set(["*"])
    
    @classmethod
    def watch_training_process(cls):
        def decorator(train_process):
            @functools.wraps(train_process)
            def wrap(*args, **kwargs):
                try:
                    result = train_process(*args, **kwargs)
                except AutoCheckpointSaverException as e:
                    raise
                except Exception as e:
                    if _is_watching_current_exceptions(cls.watching_exceptions, e.__class__.__name__):
                        _invoke_all_saver(cls.existing_savers)
                    raise
                    
                if cls.auto_detach:
                    cls.detach_saver()
                return result

            return wrap

        return decorator
    
    @classmethod
    def config(cls):
        pass
        
    @classmethod
    def attach_saver(cls, saver, name:str=None)->str:
        if name is None:
            name = _naming_saver()
        cls.existing_savers[name] = saver
        return name

    @classmethod
    def detach_saver(cls, name=None):
        if name is None:
            cls.existing_savers = {}
        else:
            if name in cls.existing_savers:
                cls.existing_savers.pop(name)

registry = CheckpointRegistry()


