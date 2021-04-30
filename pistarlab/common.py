from abc import ABCMeta, abstractmethod
# Interface
class MAEnv:

    @abstractmethod
    def reset(self):
        raise NotImplementedError()

    @abstractmethod
    def step(self, action):
        raise NotImplementedError()

    @abstractmethod
    def render(self, mode, player_id = None,*args,**kwargs):
        raise NotImplementedError()


    @abstractmethod
    def stats(self, player_id = None,*args,**kwargs):
        raise NotImplementedError()