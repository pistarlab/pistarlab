import copy
import json
import logging
import os
import pickle
import tarfile
import tempfile
import time
from typing import List

from . import ctx
from .component import Component
from .databuffer import DataBuffer
from .dbmodels import AgentModel, AgentSpecModel
from .entity import Entity
from .meta import *
from .util_funcs import *
from .utils.misc import get_timestamp_with_proc_info


class Agent(Entity):

    @staticmethod
    def load(id):
        dbmodel: AgentModel = Agent.get_dbmodel_by_id(id)
        if dbmodel is None:
            raise Exception("No such agent exists with id {}".format(id))
        if dbmodel.spec.entry_point:
            agent_cls = get_class_from_entry_point(dbmodel.spec.entry_point)
        else:
            agent_cls = Agent
        return agent_cls(
            _calling_directly=False,
            _id=dbmodel.id,
            spec_id=dbmodel.spec.id,
            config=dbmodel.config,
            meta=dbmodel.meta)

    @staticmethod
    def create(spec_id, config={}, custom_seed=None):
        spec_dbmodel: AgentSpecModel = Agent.get_spec_dbmodel_by_id(spec_id)
        if spec_dbmodel is None:
            raise Exception(
                f"No such agent spec with spec_id:{spec_id} found.")

        meta = {
            'spec_version': spec_dbmodel.version,
            'spec_extension_id': spec_dbmodel.extension_id,
            'spec_extension_version': spec_dbmodel.extension_version
        }
        if spec_dbmodel.entry_point is None:
            agent_cls = Agent
        else:
            agent_cls = get_class_from_entry_point(spec_dbmodel.entry_point)
        config = merged_dict(spec_dbmodel.config, config)
        return agent_cls(
            _calling_directly=False,
            spec_id=spec_id,
            config=config,
            meta=meta,
            custom_seed=custom_seed)

    # def create_from_snapshot(snapshot_archive_path):

    #     # TODO: load from web/external source
    #     logging.info("Loading snapshot from {}".format(snapshot_archive_path))
    #     if not os.path.exists(snapshot_archive_path):
    #         raise FileNotFoundError(
    #             "Error: Snapshot Archive not found {}".format(snapshot_archive_path))

    #     temp_dir = tempfile.mkdtemp()
    #     tar = tarfile.open(snapshot_archive_path)
    #     tar.extractall(temp_dir)
    #     tar.close()
    #     data_source_path = os.path.join(temp_dir, 'data')

    #     with open(os.path.join(data_source_path, "snapshot.json"), 'r') as f:
    #         snapshot_data = json.load(f)

    #     checkpoints_src_path = os.path.join(data_source_path, "checkpoints")
    #     spec_id = snapshot_data['spec_id']
    #     meta = snapshot_data['meta']
    #     last_checkpoint = snapshot_data['last_checkpoint']
    #     meta['source_snapshot'] = copy.deepcopy(snapshot_data)
    #     config = snapshot_data['config']
    #     notes = snapshot_data['notes']
    #     seed = snapshot_data['seed']

    #     agent: Agent = Agent.create(spec_id=spec_id, config=config)
    #     target_path = ctx.get_store().get_path_from_key(('agent', agent.get_id()))
    #     dbmodel = agent.get_dbmodel()
    #     dbmodel.meta = meta
    #     dbmodel.notes = notes
    #     dbmodel.seed = seed
    #     dbmodel.last_checkpoint = last_checkpoint
    #     import shutil
    #     shutil.copytree(checkpoints_src_path, os.path.join(
    #         target_path, "checkpoints"))
    #     ctx.get_dbsession().commit()
    #     return agent

    @staticmethod
    def get_dbmodel_by_id(id) -> AgentModel:
        query = ctx.get_dbsession().query(AgentModel)
        return query.get(id)

    @staticmethod
    def get_spec_dbmodel_by_id(id) -> AgentSpecModel:
        query = ctx.get_dbsession().query(AgentSpecModel)
        return query.get(id)

    def __init__(self, _calling_directly=True, spec_id=None, _id=None, config={}, meta={}, custom_seed=None):
        """
        Internal use only: don't use directly - use load or create

        """
        if _calling_directly:
            raise Exception(
                "Invalid instantiation method: this class should only be called from Agent.load() or Agent.create()")

        super().__init__(_id=_id, entity_type=AGENT_ENTITY)

        self._spec_id = spec_id
        self._config = config
        self._meta = meta
        self.custom_seed = custom_seed

        self._sync_data_model()
        self.stat_buffer: DataBuffer = None
        self.learn_step = None
        self.initialize()
        self.load_stats()
        self.last_flush_time = 0
        self.flush_freq_seconds = 2

    def get_id(self):
        return self._id

    def initialize(self):
        pass

    def cleanup(self):
        pass

    def __repr__(self):
        return 'id: {}, spec_id: {}'.format(self._id, self._spec_id)

    def get_seed_as_int(self):
        return int(self.get_dbmodel().seed, 0)

    def get_task_runner_cls(self):
        dbmodel: AgentModel = self.get_dbmodel()
        return get_class_from_entry_point(dbmodel.spec.runner_entry_point)

    def get_dbmodel(self) -> AgentModel:
        return Agent.get_dbmodel_by_id(self._id)

    # TODO: add lazy loading support?
    def _sync_data_model(self):
        dbmodel = self.get_dbmodel()
        if dbmodel is None:
            try:
                seed = self.custom_seed
                if seed is None:
                    seed = ctx.get_next_seed()

                logging.info("Creating new agent record. {}".format(self._id))
                dbmodel = AgentModel(
                    id=self._id,
                    config=self._config,
                    spec_id=self._spec_id,
                    seed=seed,
                    meta=self._meta)
                ctx.get_dbsession().add(dbmodel)
                ctx.get_dbsession().commit()
            except Exception as e:
                ctx.get_dbsession().rollback()
                raise e
            return True

    def get_sessions(self):
        dbmodel = self.get_dbmodel()
        ids = []
        for sess in dbmodel.sessions:
            ids.append(sess.id)
        return ids

    # Config Methods
    def get_config(self, run_config={}):
        return merged_dict(
            self.get_dbmodel().config,
            run_config)

    def update_config(self, config):
        dbmodel = self.get_dbmodel()
        dbmodel.config = copy.copy(config)
        ctx.get_dbsession().add(dbmodel)
        ctx.get_dbsession().commit()

    def update_config_key(self, key, value):
        dbmodel = self.get_dbmodel()
        config = copy.copy(dbmodel.config)
        config[key] = value
        dbmodel.config = config
        ctx.get_dbsession().add(dbmodel)
        ctx.get_dbsession().commit()

    def get_config_key(self, key, default=None):
        dbmodel = self.get_dbmodel()
        return dbmodel.config.get(key, default)

    def get_default_config(self):
        dbmodel = self.get_dbmodel()
        return dbmodel.config

    # Component Methods
    def create_components(self):
        config = self.get_config()
        components_def = config.get('components')
        if components_def is None:
            return
        for name, component_def in components_def.items():
            component_spec_id = component_def['__component_id']
            comp_config = component_def['config']
            component = Component(
                spec_id=component_spec_id, name=name,
                config=comp_config,
                agent_id=self.get_id())
            logging.info("Created Component {} with id {} for agent {}".format(
                name, component.get_id(), self.get_id()))

    def add_component(self, component: Component):
        component.set_agent(self.get_id())
        ctx.get_dbsession().commit()

    def get_component_by_name(self, name) -> Component:
        for comp in self.get_components():
            if comp.get_name() == name:
                return comp
        return None

    def get_components(self) -> List[Component]:
        dbmodel = self.get_dbmodel()
        results = []
        for item in dbmodel.components:
            results.append(Component(_id=item.id))
        return results

    # Stat Methods
    def init_stat_logger(self, cols):
        self.stat_buffer = DataBuffer(cols=cols, size=100)

    def flush_stats(self):
        if self.stat_buffer is None:
            return
        ctx.get_store().extend_multipart_dict(
            key=(AGENT_ENTITY, self._id),
            name="stats",
            value=self.stat_buffer.get_dict())
        self.stat_buffer.clear()

    def log_stat_dict(self, task_id, data):
        if self.stat_buffer is None:
            self.stat_buffer = DataBuffer(
                cols=['timestamp', 'task_id', 'learn_step'] + list(data.keys()), size=10000)
        data['timestamp'] = time.time()
        data['task_id'] = task_id
        data['learn_step'] = self.learn_step
        # data['sessions'] = sorted([sess.id for sess in self.get_dbmodel().sessions if sess.status == STATE_RUNNING])
        self.learn_step += 1
        self.stat_buffer.add_dict(data)
        if self.stat_buffer.is_full() or (time.time() - self.last_flush_time > self.flush_freq_seconds):
            self.flush_stats()

    def load_stats(self):
        dbmodel = self.get_dbmodel()
        if dbmodel.stats is None:
            self.learn_step = 0
        else:
            self.learn_step = dbmodel.stats.get('learn_step', 0)

    def save_stats(self):
        dbmodel = self.get_dbmodel()
        dbmodel.stats = {
            'learn_step': self.learn_step
        }
        ctx.get_dbsession().add(dbmodel)
        ctx.get_dbsession().commit()

    def close(self):
        if self.stat_buffer:
            self.flush_stats()

    # Checkpoint Methods
    def get_last_checkpoint(self):
        return self.get_dbmodel().last_checkpoint

    def update_last_checkpoint(self, checkpoint_id, path, meta):
        meta_file_path = os.path.join(path, "meta.json")
        with open(meta_file_path, 'w') as f:
            json.dump(meta, f)
        self.get_dbmodel().last_checkpoint = {
            'id': checkpoint_id,
            'meta': meta
        }
        ctx.get_dbsession().commit()

    def get_checkpoint_path(self, checkpoint_id):
        return ctx.get_store().get_path_from_key((self.entity_type, self.get_id(), 'checkpoints', checkpoint_id))

    def get_state(self):
        checkpoint = self.get_last_checkpoint()
        if checkpoint is None:
            return None
        else:
            path = ctx.get_store().get_path_from_key(
                (self.entity_type, self.get_id(), 'checkpoints', checkpoint['id']))
            with open(os.path.join(path, 'state.pkl'), 'rb') as f:
                return pickle.load(f)

    def save_state(self, state, meta):
        checkpoint_id = get_timestamp_with_proc_info()

        path = self.get_checkpoint_path(checkpoint_id)

        logging.info(f"Agent {self.get_id()} state saved state path:{path}")
        os.makedirs(path, exist_ok=True)

        state_file_path = os.path.join(path, "state.pkl")
        with open(state_file_path, 'wb') as f:
            pickle.dump(state, f)

        self.update_last_checkpoint(checkpoint_id, path, meta)

    def list_checkpoints(self):
        path = ctx.get_store().get_path_from_key(
            (self.entity_type, self.get_id(), 'checkpoints'))
        return os.listdir(path)
