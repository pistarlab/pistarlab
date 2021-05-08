
import json
import logging
import os
import shutil
import tempfile
import datetime
from typing import List

import pkg_resources
from redis import StrictRedis
from sqlalchemy import orm as db_orm

from .api_schema import schema
from .config import SysConfig, get_sys_config
from .data_context import DataContext
from .dbmodels import *
from .dbmodels import (AgentSpec, AgentSpecModel, EnvironmentModel, EnvSpec,
                       EnvSpecModel, SessionModel)
from .execution_context import ExecutionContext
from .meta import *
from .plugin_manager import PluginManager, create_new_plugin
from .storage import Storage
from .util_funcs import *
# from .utils.code_meta import code_src_and_meta_from_instance
from .utils.logging import (new_entity_logger, new_scoped_logger,
                            setup_default_logging, setup_logging)
from .utils.misc import gen_shortuid, generate_seed
from .utils.snapshot_helpers import make_tarfile


def get_entry_point_from_class(cls):
    return "{}:{}".format(cls.__module__, cls.__name__)


def get_task_spec_from_class(cls):
    logging.info("Registering task---------------")
    spec = {}

    spec['runner_entry_point'] = get_entry_point_from_class(cls)
    spec['entry_point'] = cls.entry_point
    spec['spec_id'] = cls.spec_id if cls.spec_id else cls.__module__
    spec['displayed_name'] = cls.displayed_name if cls.displayed_name else ""
    spec['description'] = cls.description if cls.description else ""
    spec['plugin_id'] = cls.plugin_id
    spec['plugin_version'] = cls.plugin_version

    spec['version'] = cls.version
    spec['config'] = cls.config if cls.config else {}
    return spec


PISTARLAB_INIT_KEY = "pistarlab_init"


class SysContext:

    def __init__(self):

        self._data_context: DataContext = None
        self._exec_context: ExecutionContext = None
        self.verbose = True

        self._redis_client: StrictRedis = None
        self.plugin_manager: PluginManager = None

        self._initialized = False
        self._closed = False
        self.config: SysConfig = get_sys_config()

        self.default_logger = logging

    def __del__(self):
        self.close()

    def get_data_context(self) -> DataContext:
        if self._data_context is None:
            self._data_context = DataContext(self.config)
        return self._data_context

    def get_execution_context(self) -> ExecutionContext:
        if self._exec_context is None:
            self.get_logger().info("CONFIG")
            self.get_logger().info(f"{self.config.execution_context_config}")
            self._exec_context = ExecutionContext(self.config.execution_context_config)
        return self._exec_context

    def get_logger(self):
        return self.default_logger

    def set_default_logger(self, logger):
        self.default_logger = logger

    def get_entity_logger(self, entity_type, uid, level=logging.INFO, sub_id="default"):
        return new_entity_logger(
            path=self.config.data_path,
            entity_type=entity_type,
            uid=uid,
            level=level,
            redis_client=self.get_redis_client(),
            sub_id=sub_id)

    def get_scopped_logger(self, scope_name, level=logging.INFO):
        return new_scoped_logger(path=self.config.log_root, scope_name=scope_name, level=level, redis_client=self.get_redis_client())

    def connect(self, force=False, execution_context_config=None):
        if not self._initialized or force:
            if execution_context_config is not None:
                logging.info("OVERRIDING EXECUTION CONTEXT CONFIG")
                self.config.execution_context_config = execution_context_config

            setup_default_logging(logging.DEBUG)
            logging.debug("Initializing Context")

            #########################################
            # Setup Logging
            #########################################
            LOG_LEVEL = getattr(logging, self.config.log_level.upper(), None)
            setup_logging(self.config.data_path,
                          level=LOG_LEVEL,
                          redis_client=self.get_redis_client(),
                          verbose=self.verbose)

            # Set flag
            self._initialized = True

        else:
            logging.debug("Already initalized, not reinitializing.")

    def initialize(self, force=False, execution_context_config=None):
        """
        Initializes pistarlab. Should only be on run by primary instance to avoid race
        """
        if not self._initialized or force:
            if execution_context_config is not None:
                logging.info("OVERRIDING EXECUTION CONTEXT CONFIG")
                self.config.execution_context_config = execution_context_config
            setup_default_logging(logging.DEBUG)
            logging.debug("Initializing Context")

            #########################################
            # Check if first run to avoid rerunning startup processes upon a reload
            #########################################
            first_run = False
            try:
                is_initialized = self.get_redis_client().get(PISTARLAB_INIT_KEY).decode()

                if is_initialized == "true":
                    first_run = False
                    logging.info("Already running.")
                else:
                    raise Exception("Not initialized {}".format(is_initialized))
            except Exception as e:
                logging.info("First time running pistarlab context. {}".format(e))
                self.get_redis_client().set(PISTARLAB_INIT_KEY, "true")
                first_run = True

            #########################################
            # Setup Logging
            #########################################
            LOG_LEVEL = getattr(logging, self.config.log_level.upper(), None)
            setup_logging(self.config.data_path,
                          level=LOG_LEVEL,
                          redis_client=self.get_redis_client(),
                          verbose=self.verbose)

            #########################################
            # Setup Data Context
            #########################################
            data_context = self.get_data_context()
            if first_run:
                data_context.init_db()
                data_context.cleanup()

            #########################################
            # Setup Plugins
            #########################################
            # WARNING: need to be careful with the ordering here - Don't want to create an infinite loop.
            #   ie. plugin need intiailizing contexts, which initialize plugins, which ..., etc
            self.plugin_manager = PluginManager(
                'pistarlab',
                workspace_path=self.config.workspace_path,
                data_path=self.config.data_path,
                logger=self.get_scopped_logger("plugin_manager"))
            if first_run:
                self.plugin_manager.cleanup()
                self.plugin_manager.finish_installing_new_plugins()
            self.plugin_manager.load_plugins()

            #########################################
            # Load Snapshots
            #########################################
            self.update_snapshot_index()

            # Set flag
            self._initialized = True

            # Required for Launcher
            logging.info("Backend is Ready")
        else:
            logging.debug("Already initalized, not reinitializing.")

    def create_new_plugin(self, plugin_id, plugin_name, description=""):
        create_new_plugin(
            workspace_path=self.config.workspace_path,
            author=self.get_user_id(""),
            plugin_id=plugin_id,
            plugin_name=plugin_name,
            description=description)

    def get_store(self) -> Storage:
        return self.get_data_context().get_store()

    def get_user_id(self, token):
        return self.get_data_context().get_user_id(token)

    def get_next_id(self, entity_type):
        self.get_dbsession().expire_all()
        self.get_dbsession().query(SystemCounter).filter(SystemCounter.id == entity_type).update({SystemCounter.value: SystemCounter.value + 1})
        dbmodel = self.get_dbsession().query(SystemCounter).get(entity_type)
        value = dbmodel.value
        self.get_dbsession().commit()
        prefix = ENTITY_ID_PREFIX_LOOKUP.get(entity_type, "X")
        return "{}-{}".format(prefix, value)

    def get_next_seed(self):
        return generate_seed(self.get_data_context().get_user_id(""))

    def get_best_sessions_in_for_env_spec(self, env_spec_id, summary_stat_name):
        from sqlalchemy.types import FLOAT
        sess_list = self.get_dbsession().query(SessionModel) \
            .filter(SessionModel.env_spec_id == env_spec_id) \
            .order_by(SessionModel.summary[summary_stat_name].cast(FLOAT).desc()) \
            .all()
        return sess_list

    def get_redis_client(self) -> StrictRedis:
        return self.get_data_context().get_redis_client()

    def execute_graphql(self, query, variables={}):
        self.get_dbsession().expire_all()
        return schema.execute(
            query,
            variables=variables,
            context_value={
                'session': ctx.get_dbsession(),
                'ctx': ctx})

    def get_dbsession(self) -> db_orm.Session:
        return self.get_data_context().get_dbsession()

    def close(self):
        if self._closed:
            return
        if self._data_context is not None:
            self._data_context.close()

    def get_entity_status(self, model_class, id):
        try:
            dbmodel = self.get_dbsession().query(model_class).get(id)
            return dbmodel.status
        except Exception as e:
            ctx.get_dbsession().rollback()
            raise e

    # typically, this will be handled automatically
    def modify_entity_status(self, model_class, id, state, msg=''):
        try:
            dbmodel = self.get_dbsession().query(model_class).get(id)
            dbmodel.status = state
            dbmodel.status_msg = msg
            dbmodel.status_timestamp = datetime.datetime.now()
            ctx.get_dbsession().commit()
        except Exception as e:
            ctx.get_dbsession().rollback()
            raise e
        return True

    def add_agent_tag(self, agent_id, tag_id):
        tag_id = tag_id.lower().replace(" ", "")
        try:
            model = AgentTagModel(agent_id=agent_id, tag_id=tag_id)
            session = self.get_dbsession()
            session.add(model)
            ctx.get_dbsession().commit()
        except Exception as e:
            ctx.get_dbsession().rollback()
            raise e
        return True

    def remove_agent_tag(self, agent_id, tag_id):
        tag_id = tag_id.lower().replace(" ", "")
        try:
            session = ctx.get_dbsession()
            obj = session.query(AgentTagModel).get((tag_id, agent_id))
            session.delete(obj)

            # session.query(AgentTagModel).filter(
            #     AgentTagModel.agent_id == agent_id,
            #     AgentTagModel.tag_id == tag_id).delete()
            session.commit()
        except Exception as e:
            ctx.get_dbsession().rollback()
            raise e
        return True

    # Agent specs
    def list_agent_specs(self) -> List[str]:
        query = self.get_dbsession().query(AgentSpecModel.id)
        return [v[0] for v in query.all()]

    def get_agent_spec(self, id) -> AgentSpec:
        query = self.get_dbsession().query(AgentSpecModel)
        return query.get(id)

    def get_agent_dbmodel(self, id) -> AgentModel:
        query = self.get_dbsession().query(AgentModel)
        return query.get(id)

    # Agent instances
    def list_agents(self) -> List[str]:
        query = self.get_dbsession().query(AgentModel)
        return [{'id': v.id, 'spec_id': v.spec_id, 'sessions': [s.id for s in v.sessions], 'components':[s.name for s in v.components]} for v in query.all()]

    # Component Specs
    def list_component_specs(self) -> List[str]:
        query = self.get_dbsession().query(ComponentSpecModel.id)
        return [v[0] for v in query.all()]

    def get_component_spec(self, id) -> ComponentSpecModel:
        query = self.get_dbsession().query(ComponentSpecModel)
        return query.get(id)

    # Task Specs
    def list_task_specs(self) -> List[str]:
        query = self.get_dbsession().query(TaskSpecModel.id)
        return [v[0] for v in query.all()]

    def get_task_spec(self, id) -> EnvSpec:
        query = self.get_dbsession().query(TaskSpecModel)
        return query.get(id)

    # Tasks
    def list_tasks(self) -> List[Any]:
        query = self.get_dbsession().query(TaskModel)
        return [{'id': v.id, 'spec_id': v.spec_id, 'parent_task_id': v.parent_task_id, 'status': v.status, 'sessions': [s.id for s in v.sessions], 'child_tasks':[s.id for s in v.child_tasks]} for v in query.all()]

    # Tasks
    def list_tasks_detailed(self) -> List[str]:
        query = self.get_dbsession().query(TaskModel)
        return query.all()

    # Sessions
    def list_sessions(self) -> List[str]:
        query = self.get_dbsession().query(SessionModel)
        return [{'id': v.id, 'env_spec_id': v.env_spec_id, 'task_id': v.task_id, 'agent_id': v.agent_id, 'status': v.status} for v in query.all()]

    def get_session(self, id) -> SessionModel:
        query = self.get_dbsession().query(SessionModel)
        return query.get(id)

    # Plugin methods
    def disable_plugin_by_id(self, plugin_id):
        dbsession = self.get_dbsession()
        for cls in [EnvironmentModel, AgentSpecModel, ComponentSpecModel, TaskSpecModel]:
            dbmodels = dbsession.query(cls).filter(cls.plugin_id == plugin_id).all()
            for dbmodel in dbmodels:
                dbmodel.disabled = True
        dbsession.commit()

    def list_plugins(self, status_filter=None):
        plugins = [(p['id'], p['version']) for p in filter(lambda x: status_filter is None or x['status'] == status_filter, self.plugin_manager.get_all_plugins().values())]
        return plugins

    def get_plugin(self, id, version):
        return self.plugin_manager.get_plugin(id, version)

    def install_plugin(self, id, plugin_version):
        return self.plugin_manager.install_plugin(id, plugin_version)

    def uninstall_plugin(self, id):
        return self.plugin_manager.uninstall_plugin(id)

    # Environment Specs
    def list_env_specs(self) -> List[str]:
        query = self.get_dbsession().query(EnvSpecModel.id)
        return [v[0] for v in query.all()]

    def get_env_spec(self, id) -> EnvSpec:
        query = self.get_dbsession().query(EnvSpecModel)
        return query.get(id)

    def make_env(self, spec_id):
        from .utils import env_helpers
        env = self.get_dbsession().query(EnvSpecModel).get(spec_id)
        return env_helpers.get_env_instance(env.config.get('entry_point'), kwargs=env.config.get('env_kwargs', {}))

    def install_plugin_from_manifest(self, plugin_id, plugin_version):
        plugin = self.get_plugin(plugin_id, plugin_version)
        module_name = plugin.get('module_name', plugin_id.replace("-", "_"))
        manifest_path = pkg_resources.resource_filename(module_name, "manifest.json")

        with open(manifest_path, 'r') as f:
            manifest_data = json.load(f)

        image_save_path = self.get_store().get_path_from_key(key=(SYS_CONFIG_DIR, 'envs', 'images'))
        manifest_files_path = pkg_resources.resource_filename(module_name, "manifest_files")

        try:
            for data in manifest_data.get('environments', []):
                self.register_environment(
                    plugin_id=plugin_id,
                    plugin_version=plugin_version,
                    **data)

            # ENV/ENV_SPEC
            for data in manifest_data.get('env_specs', []):
                try:
                    image_filename = data['metadata']['image_filename']
                    image_target_path = os.path.join(image_save_path, image_filename)
                    image_source_path = os.path.join(manifest_files_path, image_filename)
                    if not os.path.exists(image_target_path) and os.path.exists(image_source_path):
                        import shutil
                        shutil.copy(image_source_path, image_target_path)
                except Exception as e:
                    logging.error(f"Unable to copy image due to error while copying {e}")

                self.register_env_spec(
                    plugin_id=plugin_id,
                    plugin_version=plugin_version,
                    **data)

            # COMPONENT_SPECS
            for data in manifest_data.get('component_specs', []):
                self.register_component_spec(
                    plugin_id=plugin_id,
                    plugin_version=plugin_version,
                    **data)

            # AGENT_SPECS
            for data in manifest_data.get('agent_specs', []):
                self.register_agent_spec(
                    plugin_id=plugin_id,
                    plugin_version=plugin_version,
                    **data)

            # TASK_SPECS
            for data in manifest_data.get('task_specs', []):
                self.register_task_spec(
                    plugin_id=plugin_id,
                    plugin_version=plugin_version,
                    **data)
        except Exception as e:
            self.get_dbsession().rollback()
            raise e

    def register_env_spec_from_class(self, spec_id, env_class, *args, **kwargs):
        entry_point = get_entry_point_from_class(env_class)
        self.register_env_spec(spec_id=spec_id, entry_point=entry_point, *args, **kwargs)

    def register_environment(
            self,
            environment_id,
            default_entry_point=None,
            default_config=None,
            default_metadata=None,
            displayed_name=None,
            categories=[],
            plugin_id=None,
            plugin_version="0.0.1-dev",
            version="0.0.1-dev",
            description=None,
            disabled=False):

        session = self.get_dbsession()
        environment = session.query(EnvironmentModel).get(environment_id)
        if environment is None:
            environment = EnvironmentModel(id=environment_id)
            environment = session.merge(environment)
            session.add(environment)

        environment.displayed_name = displayed_name or environment_id
        environment.description = description
        environment.plugin_id = plugin_id
        environment.plugin_version = plugin_version
        environment.default_entry_point = default_entry_point
        environment.version = version
        environment.disabled = disabled
        environment.categories = ",".join([v.lower().replace(" ", "") for v in categories])
        environment.default_meta = default_metadata
        environment.default_config = default_config
        session.commit()

    def register_env_spec(
            self,
            spec_id,
            entry_point=None,
            env_type=RL_SINGLEPLAYER_ENV,
            tags=[],
            categories=[],
            displayed_name=None,
            environment_displayed_name=None,
            plugin_id=None,
            plugin_version="0.0.1-dev",
            version="0.0.1-dev",
            environment_id=None,
            description=None,
            config=None,
            params={},
            metadata=None,
            disabled=False):

        environment_id = environment_id or spec_id
        environment_displayed_name = environment_displayed_name or displayed_name

        session = self.get_dbsession()
        environment = session.query(EnvironmentModel).get(environment_id)
        if environment is None:
            logging.info("No Environment with name {} exists. Adding using provided values.".format(environment_id))
            environment = EnvironmentModel(id=environment_id)
            environment.displayed_name = environment_displayed_name
            environment.description = description
            environment.categories = ",".join([v.lower().replace(" ", "") for v in categories])
            environment.plugin_id = plugin_id
            environment.plugin_version = plugin_version
            environment.default_entry_point = entry_point
            environment.version = version
            environment.disabled = disabled
            environment.default_meta = metadata
            environment.default_config = config
            environment = session.merge(environment)
            session.add(environment)

        spec = session.query(EnvSpecModel).get(spec_id)
        if spec is None:
            spec = EnvSpec(id=spec_id)
            session.add(spec)

        spec.displayed_name = displayed_name or spec_id
        spec.description = description
        spec.entry_point = entry_point
        spec.meta = metadata
        spec.env_type = env_type
        spec.tags = ",".join([v.lower().replace(" ", "") for v in tags])
        spec.environment = environment
        spec.config = config
        spec.params = params
        session.commit()

    def register_agent_spec_from_classes(self, runner_cls, cls=None, *args, **kwargs):
        #TODO: merge with register_agent_spec
        if cls:
            entry_point = get_entry_point_from_class(cls)
        else:
            entry_point = None
        runner_entry_point = get_entry_point_from_class(runner_cls)
        self.register_agent_spec(
            entry_point=entry_point,
            runner_entry_point=runner_entry_point,
            *args, **kwargs)

    def register_agent_spec(
            self,
            spec_id,
            runner_entry_point,
            entry_point=None,
            config={},
            params={},
            disabled=False,
            displayed_name=None,
            plugin_id=None,
            plugin_version="0.0.1-dev",
            version="0.0.1-dev",
            description=None):

        session = self.get_dbsession()
        spec = session.query(AgentSpecModel).get(spec_id)
        if spec is None:
            spec = AgentSpecModel(id=spec_id)
            session.add(spec)

        spec.displayed_name = displayed_name or spec_id
        spec.description = description
        spec.plugin_id = plugin_id
        spec.plugin_version = plugin_version
        spec.entry_point = entry_point
        spec.runner_entry_point = runner_entry_point
        spec.version = version
        spec.disabled = disabled
        spec.config = config
        spec.params = params
        session.commit()

    def register_component_spec(
            self,
            spec_id,
            entry_point,
            parent_class_entry_point,
            config={},
            params={},
            displayed_name=None,
            plugin_id=None,
            plugin_version="0.0.1-dev",
            version="0.0.1-dev",
            category=None,
            disabled=False,
            description=None,
            metadata={}):

        session = self.get_dbsession()

        spec = session.query(ComponentSpecModel).get(spec_id)
        if spec is None:
            spec = ComponentSpecModel(id=spec_id)
            session.add(spec)

        spec.displayed_name = displayed_name or spec_id
        spec.description = description
        spec.plugin_id = plugin_id
        spec.plugin_version = plugin_version
        spec.entry_point = entry_point
        spec.parent_class_entry_point = parent_class_entry_point
        spec.config = config
        spec.params = params
        spec.version = version
        spec.disabled = disabled
        spec.meta = metadata
        spec.category = category
        session.commit()

    def register_task_spec_from_class(self, klass):
        task_spec = get_task_spec_from_class(klass)
        self.register_task_spec(**task_spec)

    def register_task_spec(
            self,
            spec_id,
            entry_point,
            runner_entry_point,
            config={},
            params={},
            displayed_name=None,
            plugin_id=None,
            plugin_version="0.0.1-dev",
            version="0.0.1-dev",
            type_name=None,
            disabled=False,
            description=None,
            metadata={}):

        session = self.get_dbsession()

        spec = session.query(TaskSpecModel).get(spec_id)
        if spec is None:
            spec = TaskSpecModel(id=spec_id)
            session.add(spec)

        spec.displayed_name = displayed_name or spec_id
        spec.description = description
        spec.plugin_id = plugin_id
        spec.plugin_version = plugin_version
        spec.entry_point = entry_point
        spec.runner_entry_point = runner_entry_point
        spec.config = config
        spec.params = params
        spec.version = version
        spec.disabled = disabled
        spec.meta = metadata
        spec.type_name = type_name
        session.commit()

    def create_agent_snapshot(
            self,
            entity_id,
            submitter_id="default",
            snapshot_description="",
            snapshot_version="0"):

        # Publish locally
        entity_type = 'agent'
        src_entity_path = self.get_store().get_path_from_key((entity_type, entity_id))
        dbmodel = self.get_agent_dbmodel(entity_id)

        spec_id = dbmodel.spec_id
        meta = dbmodel.meta
        notes = dbmodel.notes
        seed = dbmodel.seed
        config = dbmodel.config
        last_checkpoint = dbmodel.last_checkpoint
        current_timestamp = datetime.datetime.now()

        session_data = []
        for s in dbmodel.sessions:
            session_data.append({
                'env_spec_id': s.env_spec_id,
                'env_spec_version': s.env_spec_version,
                'env_spec_config': s.env_spec_config,
                'summary': s.summary})

        snapshot_data = {
            'id': entity_id,
            'seed': seed,
            'entity_type': entity_type,
            'spec_id': spec_id,
            'submitter_id': submitter_id,
            'creation_time': str(current_timestamp),
            'meta': meta,
            'notes': notes,
            'last_checkpoint': last_checkpoint,
            'snapshot_description': snapshot_description,
            'snapshot_version': snapshot_version,
            'session_data': session_data,
            'config': config}

        temp_dir = tempfile.mkdtemp()

        # Add Config
        with open(os.path.join(temp_dir, "snapshot.json"), "w") as f:
            json.dump(snapshot_data, f, indent=2)

        # Copy Checkpoint Data to dir
        checkpoints_subdir = "checkpoints"
        src_checkpoints_dir = os.path.join(src_entity_path, checkpoints_subdir)
        target_checkpoints_dir = os.path.join(temp_dir, checkpoints_subdir)
        shutil.copytree(src_checkpoints_dir, target_checkpoints_dir)

        # Push to target location
        snapshot_path = os.path.join(self.config.local_snapshot_path, entity_type, spec_id)
        os.makedirs(snapshot_path, exist_ok=True)

        # Save Data Separately as well
        snapshot_prefix = "{}__{}_v{}".format(submitter_id, seed, snapshot_version)
        with open(os.path.join(snapshot_path, f"{snapshot_prefix}.json"), 'w') as f:
            json.dump(snapshot_data, f, indent=2)

        # Save Data
        snapshot_filepath = os.path.join(snapshot_path, f"{snapshot_prefix}.tar.gz")
        make_tarfile(snapshot_filepath, temp_dir)
        self.update_snapshot_index(True)
        return snapshot_data

    def update_snapshot_index(self, force=False):
        from .utils.snapshot_helpers import get_snapshots_from_file_repo

        if not force and os.path.exists(self.config.snapshot_index_path):
            return

        # load local snapshots
        entries = get_snapshots_from_file_repo(self.config.local_snapshot_path)
        for entry in entries.values():
            entry['src'] = 'local'

        # TODO: add external snapshots

        snapshot_index = {}
        snapshot_index['entries'] = entries
        snapshot_index['creation_time'] = str(datetime.datetime.now())
        snapshot_index['sources'] = {'local': self.config.local_snapshot_path}

        with open(self.config.snapshot_index_path, 'w') as f:
            json.dump(snapshot_index, f, indent=2)

    def get_snapshot_index(self):
        self.update_snapshot_index()
        with open(self.config.snapshot_index_path, "r") as f:
            return json.load(f)

    @staticmethod
    def check_torch_status():
        import torch
        info = {}
        try:
            info['version'] = torch.__version__
            info['cuda_available'] = torch.cuda.is_available()
            gpu_count = torch.cuda.device_count()
            info['gpu_count'] = gpu_count
            if gpu_count > 0:
                info['gpu_list'] = {i: torch.cuda.get_device_name(0) for i in range(gpu_count)}
                dev_id = torch.cuda.current_device()
                info['current_gpu_device_id'] = dev_id
                info['current_gpu_device_name'] = torch.cuda.get_device_name(dev_id)
        except Exception as e:
            logging.error(e)
            info['error_messsage'] = str(e)
        return info

    @staticmethod
    def check_tensorflow_status():
        import tensorflow as tf
        from tensorflow.python.client import device_lib
        info = {}
        info['version'] = tf.__version__
        local_device_protos = device_lib.list_local_devices()
        info['gpu_list'] = {x.incarnation: x.name for x in local_device_protos if x.device_type == 'GPU'}
        info['gpu_count'] = len(info['gpu_list'])
        return info

    @staticmethod
    def get_gpu_info():
        import GPUtil
        try:
            return {gpu.id: gpu.__dict__ for gpu in GPUtil.getGPUs()}
        except Exception as e:
            info = {}
            info['error_messsage'] = str(e)
            logging.error(e)
            return info

    @staticmethod
    def tf_reset_graph():
        from tensorflow.python.framework import ops
        ops.reset_default_graph()


ctx = SysContext()
