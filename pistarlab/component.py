import json
import logging
import os
import pickle

from . import ctx
from .dbmodels import ComponentModel, ComponentSpecModel
from .entity import Entity
from .meta import *
from .util_funcs import *
from .utils.misc import get_timestamp_with_proc_info


class Component(Entity):

    @staticmethod
    def get_dbmodel_by_id(id) -> ComponentModel:
        query = ctx.get_dbsession().query(ComponentModel)
        return query.get(id)

    @staticmethod
    def get_spec_dbmodel_by_id(id) -> ComponentSpecModel:
        query = ctx.get_dbsession().query(ComponentSpecModel)
        return query.get(id)

    def __init__(self, name=None, spec_id=None, _id=None, config={}, agent_id=None):
        super().__init__(_id=_id, entity_type=COMPONENT_ENTITY)

        self._name = name
        self._spec_id = spec_id
        self._config = config
        self._agent_id = agent_id
        self._sync_data_model()

    def get_config(self):
        return self.get_dbmodel().config

    def get_name(self):
        return self.get_dbmodel().name

    def get_spec_id(self):
        return self.get_dbmodel().spec_id

    def get_spec(self):
        return self.get_dbmodel().spec

    def get_type(self):
        return self.get_spec_dbmodel_by_id(self.get_spec_id()).category

    def get_last_checkpoint(self):
        return self.get_dbmodel().last_checkpoint

    def get_state(self):
        checkpoint = self.get_last_checkpoint()
        if checkpoint is None:
            return None
        else:
            path = ctx.get_store().get_path_from_key((self.entity_type, self.get_id(), 'checkpoints'))
            with open("{}.pkl".format(os.path.join(path, checkpoint['id'])), 'rb') as f:
                return pickle.load(f)

    def save_state(self, state, meta):
        checkpoint_id = get_timestamp_with_proc_info()
        self.get_dbmodel().last_checkpoint = {
            'id': checkpoint_id,
            'meta': meta
        }
        path = ctx.get_store().get_path_from_key((self.entity_type, self.get_id(), 'checkpoints'))

        logging.info(f"Saved state to:{path}")
        os.makedirs(path, exist_ok=True)

        state_file_path = "{}.pkl".format(os.path.join(path, checkpoint_id))
        with open(state_file_path, 'wb') as f:
            pickle.dump(state, f)
        meta_file_path = "{}_meta.json".format(os.path.join(path, checkpoint_id))
        with open(meta_file_path, 'w') as f:
            json.dump(meta, f)
        ctx.get_dbsession().commit()

    def get_component_class(self):
        return get_class_from_entry_point(self.get_dbmodel().spec.entry_point)

    def get_dbmodel(self) -> ComponentModel:
        return Component.get_dbmodel_by_id(self._id)

    # TODO: add lazy loading support?
    def _sync_data_model(self):
        dbmodel = self.get_dbmodel()
        if dbmodel is None:
            try:
                logging.info("Creating new component record. {}".format(self._id))
                spec_dbmodel: ComponentSpecModel = Component.get_spec_dbmodel_by_id(self._spec_id)
                self._config = merged_dict(spec_dbmodel.config, self._config)
                dbmodel = ComponentModel(
                    id=self._id,
                    name=self._name,
                    agent_id=self._agent_id,
                    config=self._config,
                    spec_id=self._spec_id)
                ctx.get_dbsession().add(dbmodel)
                ctx.get_dbsession().commit()
            except Exception as e:
                ctx.get_dbsession().rollback()
                raise e
            return True

    def get_child_component_ids(self):
        dbmodel = self.get_dbmodel()
        ids = []
        for item in dbmodel.child_components:
            ids.append(item.id)
        return ids

    def get_agent_id(self):
        dbmodel = self.get_dbmodel()
        if dbmodel.agent_id is None:
            return None
        else:
            return dbmodel.agent_id

    def set_agent(self, agent_id):
        self.get_dbmodel().agent_id = agent_id
        ctx.get_dbsession().commit()

    def get_id(self):
        return self._id

    def __repr__(self):
        return 'Component: id: {}, name: {}, spec_id: {}, type: {}'.format(self._id, self.get_name(), self.get_spec_id(), self.get_type())
