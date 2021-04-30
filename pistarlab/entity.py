from abc import abstractmethod
from typing import Any, Dict

from . import ctx



class Entity:

    def __init__(self,
                 _id: str = None,
                 entity_type=None):
        self._id = _id or ctx.get_next_id(entity_type)
        self.entity_type = entity_type

    def get_id(self) -> str:
        return self._id

    def get_entity_type(self):
        return self.entity_type

    def store_data(self, name, data):
        ctx.get_store().save(key=(self.entity_type, self.get_id), name=name, value=data)

    def get_config(self):
        return self.get_dbmodel().config

    @abstractmethod
    def get_dbmodel(self) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def _sync_data_model(self):
        raise NotImplementedError()
