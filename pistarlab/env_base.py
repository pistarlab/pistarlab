

from collections.abc import Iterable

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