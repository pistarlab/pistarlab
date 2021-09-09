

from abc import  abstractmethod, abstractproperty
from collections.abc import Iterable
from typing import Tuple,Dict,Any
from gym.spaces import Space
class DataEnvironment:


    def __init__(self):
        pass

    def get_data_filepath(self)->str:
        return

    def get_eval_filepath(self)->str:
        return

    def get_data_loader(self)->Iterable:
        return 

    def get_eval_loader(self)->Iterable:
        return

    def evaluate(self,expected,predicted):
        return

    def submit_predictions(self,outputs,predictions):
        return
        
    def render_inputs(self, inputs):
        pass

    def render_outputs(self,output):
        pass

    def close(self):
        return

class MARLEnvironment:
    """
    TODO: Finish and use this
    """

    def __init__(self):
        pass

    @abstractproperty
    def players(self):
        return

    @abstractproperty
    def possible_players(self):
        return

    @abstractproperty
    def num_players(self):
        return

    @abstractproperty
    def max_players(self):
        return

    @abstractproperty
    def min_players(self):
        return

    @abstractproperty
    def observation_spaces(self)->Dict[str,Space]:
        return

    @abstractproperty
    def action_spaces(self)->Dict[str,Space]:
        return

    @abstractmethod
    def reset(self):
        return

    @abstractmethod
    def step(self,action_dict):
        return

    @abstractmethod
    def render(self,*args,**kwargs):
        return

    @abstractmethod
    def close(self):
        return