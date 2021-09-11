
import json
import logging
import os
import shutil
import tempfile
import datetime
import requests
import base64
from unittest.case import skip
import requests
from typing import List, Any
import time
import pkg_resources
from redis import StrictRedis
from sqlalchemy import orm as db_orm
from sqlalchemy.orm.base import NON_PERSISTENT_OK

from .api_schema import schema
from .config import SysConfig, get_sys_config
from .data_context import DataContext
from .dbmodels import *
from .dbmodels import (AgentSpec, AgentSpecModel, EnvironmentModel, EnvSpec,
                       EnvSpecModel, SessionModel)
from .execution_context import ExecutionContext
from .meta import *
from .extension_manager import ExtensionManager, create_new_extension
from .storage import Storage
from .util_funcs import *
# from .utils.code_meta import code_src_and_meta_from_instance
from .utils.logging import (new_entity_logger, new_scoped_logger,
                            setup_default_logging, setup_logging)
from .utils.misc import gen_shortuid, generate_seed, gen_uuid
from .utils.snapshot_helpers import make_tarfile, extract_tarfile
from .utils.env_helpers import get_wrapped_env_instance


from . import __version__


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
    spec['extension_id'] = cls.extension_id
    spec['extension_version'] = cls.extension_version

    spec['version'] = cls.version
    spec['config'] = cls.config if cls.config else {}
    return spec


def get_display_info(config: SysConfig):
    try:
        with open(os.path.join(config.root_path, "display_info.json"), "r") as f:
            display_info = json.load(f)
        display_id = display_info.get("id")
        if display_id is not None:
            logging.info(
                f"Change Display :{os.environ.get('DISPLAY')} to :{display_id}")
            os.environ['DISPLAY'] = f":{display_id}"
        else:
            logging.error("Display ID not found")
        return display_info
    except Exception as e:
        logging.debug(f"ERROR loading display info {e}")
        return {}


PISTARLAB_INIT_KEY = "pistarlab_init"


class SysContext:
    """
    TODO: Bloated, need to move many(most) functions to their own logical location
    """

    def __init__(self):

        self._data_context: DataContext = None
        self._exec_context: ExecutionContext = None
        self.verbose = True

        self._redis_client: StrictRedis = None
        self.extension_manager: ExtensionManager = None

        self._initialized = False
        self._closed = False
        self.config: SysConfig = get_sys_config()
        self.default_logger = logging

        self.display_info = get_display_info(self.config)
        self.test_mode = os.environ.get("PISTARLAB_DEV_MODE") == True

        self.auth_client_id = "q3qohs4ii6hl3t0sd4dts57k6"
        self.auth_client_secret = "119i0aajl2kv5trs4vipn83c6drpulp4len473tgnlakb0gg6fck"
        self.auth_grant_type = 'authorization_code'
        self.auth_user_info_uri = "https://pistarai.auth.us-east-1.amazoncognito.com/oauth2/userInfo"
        self.auth_redirect_uri = "http://localhost:7777/api/auth"
        self.auth_redirect_logout_uri = "http://localhost:7777/api/logout"
        self.auth_token_uri = "https://pistarai.auth.us-east-1.amazoncognito.com/oauth2/token"
        self.login_uri = f"https://pistarai.auth.us-east-1.amazoncognito.com/login?client_id={self.auth_client_id}&response_type=code&scope=aws.cognito.signin.user.admin+email+openid+phone+profile&redirect_uri={self.auth_redirect_uri}"
        self.logout_uri = f"https://pistarai.auth.us-east-1.amazoncognito.com/logout?client_id={self.auth_client_id}&logout_uri={self.auth_redirect_logout_uri}"

        if self.test_mode:
            logging.info("RUNNING IN TEST MODE")
            self.cloud_api_uri = "http://127.0.0.1:3000"
        else:
            self.cloud_api_uri = "https://api.pistar.ai/Prod/v0.1"

    def __del__(self):
        self.close()

    def get_data_context(self) -> DataContext:
        if self._data_context is None:
            self._data_context = DataContext(self.config)
        return self._data_context

    def get_execution_context(self) -> ExecutionContext:
        if self._exec_context is None:
            self._exec_context = ExecutionContext(
                self.config.execution_context_config)
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

    def start_minimal_mode(self):
        import pistarlab.launcher as launcher
        launcher.start_minimal_mode()

    def stop_minimal_mode(self):
        import pistarlab.launcher as launcher
        launcher.stop_minimal_mode()

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

            self.extension_manager = ExtensionManager(
                'pistarlab',
                workspace_path=self.config.workspace_path,
                data_path=self.config.data_path,
                logger=self.get_scopped_logger("extension_manager"))
            self.extension_manager.load_extensions()

            # Set flag
            self._initialized = True

        else:
            logging.debug("Already initalized, not reinitializing.")

    def initialize(self, force=False, execution_context_config=None, log_mode=logging.INFO):
        """
        Initializes pistarlab. Should only be on run by primary instance to avoid race
        """
        if not self._initialized or force:
            if execution_context_config is not None:
                logging.info("OVERRIDING EXECUTION CONTEXT CONFIG")
                self.config.execution_context_config = execution_context_config
            setup_default_logging(log_mode)
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
                    raise Exception(
                        "Not initialized {}".format(is_initialized))
            except Exception as e:
                logging.info("First time running pistarlab context.")
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
            # Setup Extensions
            #########################################
            # WARNING: need to be careful with the ordering here - Don't want to create an infinite loop.
            #   ie. extension need intiailizing contexts, which initialize extensions, which ..., etc
            self.extension_manager = ExtensionManager(
                'pistarlab',
                workspace_path=self.config.workspace_path,
                data_path=self.config.data_path,
                logger=self.get_scopped_logger("extension_manager"))
            if first_run:
                self.extension_manager.cleanup()
                self.extension_manager.finish_installing_new_extensions()
            self.extension_manager.load_extensions()

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

    def create_new_extension(self, extension_id, extension_name, description=""):
        create_new_extension(
            workspace_path=self.config.workspace_path,
            extension_id=extension_id,
            extension_name=extension_name,
            description=description,
            original_author=self.get_user_id(),
            extension_author=self.get_user_id(),)

    def get_launcher_info(self) -> Dict[str, Any]:
        with open(os.path.join(self.config.root_path, ".launcher_runtime_settings.json"), 'r') as f:
            return json.load(f)

    def get_workspace_info(self):
        extensions = self.extension_manager.get_all_extensions()
        extensions = [extension for pid, extension in extensions.items(
        ) if extension["source"]["type"] == "workspace"]
        return {'path': self.config.workspace_path, 'extensions': extensions}

    def get_store(self) -> Storage:
        return self.get_data_context().get_store()

    def get_user_id(self):
        if self.test_mode:
            return "testuser"
        else:
            return self.get_data_context().get_user_id()

    def get_user_info(self):
        return self.get_data_context().get_user_info()

    def get_next_id(self, entity_type):
        self.get_dbsession().expire_all()
        self.get_dbsession().query(SystemCounter).filter(SystemCounter.id ==
                                                         entity_type).update({SystemCounter.value: SystemCounter.value + 1})
        dbmodel = self.get_dbsession().query(SystemCounter).get(entity_type)
        value = dbmodel.value
        self.get_dbsession().commit()
        prefix = ENTITY_ID_PREFIX_LOOKUP.get(entity_type, "X")
        return "{}-{}".format(prefix, value)

    def get_next_seed(self):
        return generate_seed(self.get_data_context().get_user_id())

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

    #########################################
    # AUTHENTICATION FUNCTIONS
    #########################################

    def _get_token_data(self, code):

        message = bytes(
            f"{self.auth_client_id}:{self.auth_client_secret}", 'utf-8')
        secret_hash = base64.b64encode(message).decode()
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": f"Basic {secret_hash}"}
        payload = f"""grant_type={self.auth_grant_type}&client_id={self.auth_client_id}&code={code}&redirect_uri={self.auth_redirect_uri}"""

        resp = requests.post(self.auth_token_uri,
                             headers=headers, data=payload)
        return resp.json()

    def logout(self, response_data={}, auth_state="logout"):
        user_info = self.get_user_info()
        user_info["last_auth_state"] = auth_state
        user_info['last_update'] = time.time()
        user_info['response_data'] = response_data
        user_info['token_data'] = None
        self.get_data_context().update_user_info(user_info)

    def _save_auth_state(self, token_data):
        """
        Persists authentication and user info
        """
        access_token = token_data['access_token']
        authorization = f"Bearer {access_token}"
        try:
            r = requests.get(self.auth_user_info_uri, headers={
                             'Authorization': authorization})
            result = r.json()
        except Exception as e:
            result = {'error': 'exception', 'error_description': str(e)}

        if "error" in result:
            logging.info("Login failed: Updating user info")
            self.logout(result, auth_state="failed")
            raise Exception(f"Failed to retrieve user info:{result}")

        else:
            logging.info("Login successful: Updating user info")
            user_info = self.get_user_info()
            user_info['user_id'] = result['username']
            user_info["last_auth_state"] = "success"
            user_info["last_update"] = time.time()
            user_info['response_data'] = result
            user_info['token_data'] = token_data
            self.get_data_context().update_user_info(user_info)
            return user_info

    def refresh_auth(self):

        user_info = self.get_user_info()
        token_data = user_info.get('token_data')
        if token_data is None or "refresh_token" not in token_data:
            raise Exception(f"No refresh token available.")

        message = bytes(
            f"{self.auth_client_id}:{self.auth_client_secret}", 'utf-8')
        secret_hash = base64.b64encode(message).decode()
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": f"Basic {secret_hash}"}
        payload = f"""grant_type=refresh_token&client_id={self.auth_client_id}&refresh_token={token_data.get('refresh_token')}"""

        resp = requests.post(self.auth_token_uri,
                             headers=headers, data=payload)
        result = resp.json()

        if "error" in result:
            self.logout(result, auth_state="failed")
            raise Exception(f"Unable to refresh access token: {result}")
        return self._save_auth_state(result)

    def update_auth(self, code):
        result = self._get_token_data(code)
        if "error" in result:
            raise Exception(f"Unable to retrieve access token: {result}")
        else:
            return self._save_auth_state(result)

    def is_logged_in(self):
        try:
            self.get_auth_token()
            return True
        except Exception as e:
            return False

    def get_auth_token(self, token_name='access_token'):
        user_info = self.get_user_info()
        token_data = user_info.get('token_data')
        if token_data is None:
            raise Exception("Not logged in. No token data found")

        if (time.time() - user_info.get('last_update', 0)) >= token_data.get('expires_in', 0):
            user_info = self.refresh_auth()
            token_data = user_info.get('token_data')

        return token_data[token_name]

    #########################################
    # STATUS FUNCTIONS
    #########################################
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

    #########################################
    # AGENT FUNCTIONS
    #########################################
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
        query = self.get_dbsession().query(AgentSpecModel)
        return [v for v in query.all() if v.disabled == False]

    def list_agent_spec_ids(self) -> List[str]:
        return [v.id for v in self.list_agent_specs() if v.disabled == False]

    def get_agent_spec(self, id) -> AgentSpec:
        query = self.get_dbsession().query(AgentSpecModel)
        return query.get(id)

    def get_agent_dbmodel(self, id) -> AgentModel:
        query = self.get_dbsession().query(AgentModel)
        return query.get(id)

    # Agent instances
    def list_agents(self, include_archive=False) -> List[str]:
        query = self.get_dbsession().query(AgentModel)
        return [v.id for v in query.all() if include_archive or v.archived is False]

    def list_agents_detailed(self) -> List[str]:
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
    def list_tasks(self, status_filter=None) -> List[Any]:
        query = self.get_dbsession().query(TaskModel)
        if status_filter is not None:
            tasks = [v for v in query.all() if v.status in status_filter]
        else:
            tasks = query.all()
        return [{'id': v.id, 'spec_id': v.spec_id, 'parent_task_id': v.parent_task_id, 'status': v.status, 'sessions': [s.id for s in v.sessions], 'child_tasks':[s.id for s in v.child_tasks]} for v in tasks]

    # Tasks
    def list_tasks_detailed(self) -> List[str]:
        query = self.get_dbsession().query(TaskModel)
        return query.all()

    # Sessions
    def list_sessions(self, status_filter=None) -> List[str]:
        query = self.get_dbsession().query(SessionModel)
        sessions = [v for v in query.all() if v.archived is False]
        if status_filter is not None:
            filtered_sessions = [
                s for s in sessions if s.status in status_filter]
            return filtered_sessions
        else:
            return sessions

    def list_sessions_detailed(self, status_filter=None) -> List[str]:
        query = self.get_dbsession().query(SessionModel)
        sessions = [{'id': v.id, 'env_spec_id': v.env_spec_id, 'task_id': v.task_id,
                     'agent_id': v.agent_id, 'status': v.status} for v in query.all()]
        if status_filter is not None:
            filtered_sessions = [
                s for s in sessions if s['status'] in status_filter]
            return filtered_sessions
        else:
            return sessions

    def get_session(self, id) -> SessionModel:
        query = self.get_dbsession().query(SessionModel)
        return query.get(id)

    # Extension methods
    def disable_extension_by_id(self, extension_id):
        dbsession = self.get_dbsession()
        for cls in [EnvironmentModel, AgentSpecModel, ComponentSpecModel, TaskSpecModel]:
            dbmodels = dbsession.query(cls).filter(
                cls.extension_id == extension_id).all()
            for dbmodel in dbmodels:
                dbmodel.disabled = True
        dbsession.commit()

    def list_extensions(self, status_filter=None):
        extensions = [(p['id'], p['version']) for p in filter(lambda x: status_filter is None or x['status']
                                                              == status_filter, self.extension_manager.get_all_extensions().values())]
        return extensions

    def get_extension(self, id, version):
        return self.extension_manager.get_extension(id, version)

    def install_extension(self, id, extension_version):
        return self.extension_manager.install_extension(id, extension_version)

    def uninstall_extension(self, id):
        return self.extension_manager.uninstall_extension(id)

    # Environments
    def list_environments(self) -> List[str]:
        query = self.get_dbsession().query(EnvironmentModel)
        return [v for v in query.all() if v.disabled == False]

    def list_environment_ids(self) -> List[str]:
        return [v.id for v in self.list_environments()]

    def get_environment(self, id) -> EnvironmentModel:
        query = self.get_dbsession().query(EnvironmentModel)
        return query.get(id)

    # Environment Specs

    def list_env_specs(self) -> List[str]:
        query = self.get_dbsession().query(EnvSpecModel)
        return [v for v in query.all() if v.environment.disabled == False]

    def list_env_spec_ids(self) -> List[str]:
        return [v.id for v in self.list_env_specs()]

    def get_env_spec(self, id) -> EnvSpec:
        query = self.get_dbsession().query(EnvSpecModel)
        return query.get(id)

    def get_env_spec_instance(self, spec_id, env_kwargs={}):
        spec = self.get_env_spec(spec_id)
        init_env_kwargs = spec.config['env_kwargs']
        kwargs = merged_dict(init_env_kwargs, env_kwargs)
        default_wrappers = spec.config['default_wrappers']
        return get_wrapped_env_instance(
            entry_point=spec.entry_point,
            kwargs=kwargs,
            wrappers=default_wrappers)

    def make_env(self, spec_id):
        from .utils import env_helpers
        env = self.get_dbsession().query(EnvSpecModel).get(spec_id)
        return env_helpers.get_env_instance(env.config.get('entry_point'), kwargs=env.config.get('env_kwargs', {}))

    def install_extension_from_manifest(
            self,
            extension_id,
            extension_version,
            replace_images=True):
        extension = self.get_extension(extension_id, extension_version)
        module_name = extension.get(
            'module_name', extension_id.replace("-", "_"))
        manifest_path = pkg_resources.resource_filename(
            module_name, "manifest.json")
        self.get_logger().info("Loading extension manifest files from {}".format(manifest_path))

        with open(manifest_path, 'r') as f:
            manifest_data = json.load(f)

        manifest_files_path = pkg_resources.resource_filename(
            module_name, "manifest_files")
        self.get_logger().info(
            "Loading extension manifest files from {}".format(manifest_files_path))

        try:
            for data in manifest_data.get('environments', []):

                self.register_environment(
                    extension_id=extension_id,
                    extension_version=extension_version,
                    manifest_files_path=manifest_files_path,
                    **data)

            # image_save_path = self.get_store().get_path_from_key(
            # key=(SYS_CONFIG_DIR, 'envs', 'images'))

            # # ENV/ENV_SPEC
            # for data in manifest_data.get('env_specs', []):
            #     self.get_logger().info(f"{data}")
            #     try:
            #         image_filename = data['metadata']['image_filename']
            #         image_target_path = os.path.join(
            #             image_save_path, image_filename)
            #         image_source_path = os.path.join(
            #             manifest_files_path, image_filename)
            #         if (not os.path.exists(image_target_path) or replace_images) and os.path.exists(image_source_path):
            #             import shutil
            #             self.get_logger().info(f"Copying spec image from {image_source_path} to {image_target_path}")
            #             shutil.copy(image_source_path, image_target_path)
            #     except Exception as e:
            #         logging.error(
            #             f"Unable to copy image due to error while copying {e}")

            #     self.register_env_spec_and_environment(
            #         extension_id=extension_id,
            #         extension_version=extension_version,
            #         **data)

            # COMPONENT_SPECS
            for data in manifest_data.get('component_specs', []):
                self.register_component_spec(
                    extension_id=extension_id,
                    extension_version=extension_version,
                    **data)

            # AGENT_SPECS
            for data in manifest_data.get('agent_specs', []):
                self.register_agent_spec(
                    extension_id=extension_id,
                    extension_version=extension_version,
                    **data)

            # TASK_SPECS
            for data in manifest_data.get('task_specs', []):
                self.register_task_spec(
                    extension_id=extension_id,
                    extension_version=extension_version,
                    **data)
        except Exception as e:
            self.get_dbsession().rollback()
            raise e

    def register_env_spec_from_class(self, spec_id, env_class, *args, **kwargs):
        entry_point = get_entry_point_from_class(env_class)
        self.register_env_spec_and_environment(
            spec_id=spec_id, entry_point=entry_point, *args, **kwargs)

    def copy_file(self, source, target, replace_files=True):
        if (not os.path.exists(target) or replace_files):
            import shutil
            self.get_logger().info(
                f"Copying spec image from {source} to {target}")
            shutil.copy(source, target)

    def register_environment(
            self,
            environment_id,
            default_entry_point=None,
            default_config=None,
            default_meta=None,
            displayed_name=None,
            categories=[],
            collection=None,
            extension_id=None,
            extension_version="0.0.1.dev0",
            version="0.0.1.dev0",
            description=None,
            usage=None,
            disabled=False,
            env_specs=None,
            skip_commit=False,
            manifest_files_path=None,
            replace_images=True):
        # NOTE: If updated also update, env_helpers.get_environment_data
        self.get_logger().info(f"importing environment_id {environment_id}")
        session = self.get_dbsession()
        environment = session.query(EnvironmentModel).get(environment_id)
        if environment is None:
            environment = EnvironmentModel(id=environment_id)
            # environment = session.merge(environment)
            session.add(environment)

        environment.displayed_name = displayed_name or environment_id
        environment.description = description
        environment.extension_id = extension_id
        environment.extension_version = extension_version
        environment.default_entry_point = default_entry_point
        environment.version = version
        environment.collection = collection
        environment.usage = usage
        environment.disabled = disabled
        environment.categories = ",".join(
            [v.lower().replace(" ", "") for v in categories])
        environment.default_meta = default_meta
        environment.default_config = default_config
        if not skip_commit:
            session.commit()

        image_save_path = self.get_store().get_path_from_key(
            key=(SYS_CONFIG_DIR, 'envs', 'images'))

        env_image_created = False
        env_image_target_path = os.path.join(
            image_save_path, f"env_{environment_id}.jpg")
        if manifest_files_path is None:
            env_image_source_path = pkg_resources.resource_filename(
                "pistarlab", "templates/env_default.jpg")
        else:
            env_image_source_path = os.path.join(
                manifest_files_path, f"env_{environment_id}.jpg")

        try:
            self.copy_file(
                env_image_source_path,
                env_image_target_path,
                replace_files=replace_images)
            env_image_created = True
        except Exception as e:
            env_image_created = False
            self.get_logger().error(
                f"Unable to copy image due to error while copying {e}")

        if env_specs is not None:
            for data in env_specs:
                spec_id = data['spec_id']
                self.get_logger().info(f"importing spec_id {spec_id}")
                self.register_env_spec(
                    environment_id=environment_id,
                    manifest_files_path=manifest_files_path,
                    replace_images=replace_images,
                    **data)
                # Use spec image for environment
                if manifest_files_path is not None:
                    image_source_path = os.path.join(
                        manifest_files_path, f"{spec_id}.jpg")
                    if not env_image_created and os.path.exists(image_source_path):
                        self.get_logger().info(
                            f"Environment image not found, using EnvSpec: {spec_id} image for Environment: {environment_id} image")
                        self.copy_file(
                            image_source_path,
                            env_image_target_path,
                            replace_files=replace_images)
                        env_image_created = True
        # if not env_image_created:
        #     default_image_path = pkg_resources.resource_filename(
        #     "pistarlab", "templates/env_default.jpg")
        #     self.copy_file(
        #         default_image_path,
        #         env_image_target_path,
        #         replace_files=replace_images)

    def register_env_spec(
            self,
            spec_id=None,
            environment_id=None,
            entry_point=None,
            human_entry_point=None,
            human_config={},
            human_description=None,
            env_type=RL_SINGLEPLAYER_ENV,
            tags=[],
            displayed_name=None,
            spec_displayed_name=None,
            description=None,
            usage=None,
            config=None,
            params={},
            metadata=None,
            skip_commit=False,
            manifest_files_path=None,
            replace_images=True):
        """
        NOTE: If updated also update, env_helpers.get_env_spec_data
        """
        environment_id = environment_id or spec_id
        spec_displayed_name = spec_displayed_name or displayed_name

        session = self.get_dbsession()
        environment = session.query(EnvironmentModel).get(environment_id)
        if environment is None:
            msg = f"No Environment with name {environment_id} exists. Adding using provided values."
            raise Exception(msg)

        spec = session.query(EnvSpecModel).get(spec_id)
        if spec is None:
            spec = EnvSpec(id=spec_id)
            session.add(spec)

        spec.environment_id = environment_id  # parent relationship
        spec.displayed_name = displayed_name or spec_id
        spec.spec_displayed_name = spec_displayed_name
        spec.description = description
        spec.usage = usage
        spec.entry_point = entry_point
        spec.human_entry_point = human_entry_point
        spec.human_config = human_config
        spec.human_description = human_description
        
        spec.meta = metadata
        spec.env_type = env_type
        spec.tags = ",".join([v.lower().replace(" ", "") for v in tags])
        spec.config = config
        spec.params = params
        if not skip_commit:
            session.commit()
        image_save_path = self.get_store().get_path_from_key(
            key=(SYS_CONFIG_DIR, 'envs', 'images'))
        image_target_path = os.path.join(image_save_path, f"{spec_id}.jpg")
        image_source_path = os.path.join(
            manifest_files_path, f"{spec_id}.jpg")
        if manifest_files_path is None or not os.path.exists(image_source_path):
            image_source_path = pkg_resources.resource_filename(
                "pistarlab", "templates/env_default.jpg")

        self.copy_file(
            image_source_path,
            image_target_path,
            replace_files=replace_images)

    def register_env_spec_and_environment(
            self,
            spec_id,
            entry_point=None,
            env_type=RL_SINGLEPLAYER_ENV,
            tags=[],
            categories=[],
            displayed_name=None,
            spec_displayed_name=None,
            environment_displayed_name=None,
            extension_id=None,
            extension_version="0.0.1.dev0",
            version="0.0.1.dev0",
            environment_id=None,
            collection=None,
            description=None,
            usage=None,
            config=None,
            params={},
            metadata=None,
            disabled=False):
        # NOTE: If updated also update, env_helpers.get_env_spec_data
        environment_id = environment_id or spec_id
        environment_displayed_name = environment_displayed_name or displayed_name
        spec_displayed_name = spec_displayed_name or displayed_name

        self.register_environment(
            environment_id=environment_id,
            default_entry_point=entry_point,
            default_config=config,
            default_meta=metadata,
            collection=collection,
            displayed_name=environment_displayed_name,
            categories=categories,
            extension_id=extension_id,
            extension_version=extension_version,
            version=version,
            description=description,
            usage=usage,
            disabled=disabled,
            skip_commit=True
        )

        self.register_env_spec(
            spec_id,
            environment_id=environment_id,
            displayed_name=displayed_name,
            spec_displayed_name=spec_displayed_name,
            description=description,
            usage=usage,
            entry_point=entry_point,
            metadata=metadata,
            env_type=env_type,
            tags=tags,
            config=config,
            params=params,
        )

    def register_agent_spec_from_classes(self, runner_cls, cls=None, *args, **kwargs):
        # TODO: merge with register_agent_spec
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
            algo_type_ids=[],
            displayed_name=None,
            collection=None,
            extension_id=None,
            extension_version="0.0.1.dev0",
            version="0.0.1.dev0",
            description=None,
            usage=None):
        """
        NOTE: must be mainted with  agent_helpers.get_agent_spec_data
        """
        session = self.get_dbsession()
        spec = session.query(AgentSpecModel).get(spec_id)
        if spec is None:
            spec = AgentSpecModel(id=spec_id)
            session.add(spec)

        spec.displayed_name = displayed_name or spec_id
        spec.description = description
        spec.usage = usage
        spec.extension_id = extension_id
        spec.extension_version = extension_version
        spec.entry_point = entry_point
        spec.runner_entry_point = runner_entry_point
        spec.version = version
        spec.collection = collection
        spec.algo_type_ids = ",".join(algo_type_ids) if algo_type_ids else None
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
            extension_id=None,
            extension_version="0.0.1.dev0",
            version="0.0.1.dev0",
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
        spec.extension_id = extension_id
        spec.extension_version = extension_version
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
            extension_id=None,
            extension_version="0.0.1.dev0",
            version="0.0.1.dev0",
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
        spec.extension_id = extension_id
        spec.extension_version = extension_version
        spec.entry_point = entry_point
        spec.runner_entry_point = runner_entry_point
        spec.config = config
        spec.params = params
        spec.version = version
        spec.disabled = disabled
        spec.meta = metadata
        spec.type_name = type_name
        session.commit()

    def load_remote_snapshot_index(self, url):
        import urllib
        index_url = os.path.join(url, 'index.json')
        with urllib.request.urlopen(index_url) as url:
            snapshot_index = json.loads(url.read().decode())

        return snapshot_index.get('entries')

    def update_snapshot_index(self, force=True):
        from .utils.snapshot_helpers import get_snapshots_from_file_repo

        if not force and os.path.exists(self.config.snapshot_index_path):
            return

        # TODO: This code is not being used
        # main_remote_url = "https://raw.githubusercontent.com/pistarlab/pistarlab-repo/main/snapshots/"
        # try:
        #     entries = self.load_remote_snapshot_index(main_remote_url)
        # except Exception as e:
        #     self.get_logger().error(
        #         f"Unable to load remote snapshot index {main_remote_url} \n {e}")
        #     entries = {}

        # entries = {}
        # for entry in entries.values():
        #     entry['src'] = 'main'
        # main_remote_url_repo = os.path.join(main_remote_url, "repo")
        entries = {}
        # load local snapshots
        local_entries = get_snapshots_from_file_repo(
            self.config.local_snapshot_path)
        entries.update(local_entries)

        snapshot_index = {}
        snapshot_index['entries'] = entries
        snapshot_index['creation_time'] = str(datetime.datetime.now())
        snapshot_index['sources'] = {
            'local': self.config.local_snapshot_path
            # 'main': main_remote_url_repo
        }

        with open(self.config.snapshot_index_path, 'w') as f:
            json.dump(snapshot_index, f, indent=2)

    def get_snapshot_index(self):
        self.update_snapshot_index()
        with open(self.config.snapshot_index_path, "r") as f:
            return json.load(f)

    def clone_agent(
            self,
            agent_id):

        snapshot_data = self.create_agent_snapshot(
            agent_id=agent_id,
            snapshot_description=f"Agent Clone of {agent_id}",
            snapshot_version=f"PRECLONE_{time.time()}",
            note_addition=f"Agent Clone of {agent_id}")
        agent = self.create_agent_from_snapshot(snapshot_data['snapshot_id'])
        return agent

    def publish_snapshot(self, snapshot_id, public=True):
        snapshot_index = self.get_snapshot_index()
        snapshot_data = snapshot_index['entries'].get(snapshot_id, None)
        snapshot_archive_path = "{}.tar.gz".format(os.path.join(
            self.config.local_snapshot_path, snapshot_data['path'], snapshot_data['file_prefix']))

        spec_id = snapshot_data['spec_id']
        agent_id = snapshot_data['id']
        agent_name = snapshot_data['agent_name']
        version = snapshot_data['snapshot_version']
        description = snapshot_data['snapshot_description']

        if agent_name is None:
            from pistarlab.agent import Agent
            agent = Agent.load(agent_id)
            agent.update_name(agent_id)
            agent_name = agent_id

        # Request upload URL
        publish_data = {
            # 'user_id': self.get_user_id(),
            'snapshot_id': snapshot_id,
            'seed': snapshot_data['seed'],
            'agent_id': agent_id,
            'agent_name': agent_name,
            'spec_id': spec_id,
            'version': version,
            'public': public,
            'description': description,
            'lab_version': __version__,
            'snapshot_data': snapshot_data
        }
        logging.info(f"Snapshot Data {snapshot_data}")
        logging.info(f"Publish Data {publish_data}")
        pub_url = f'{self.cloud_api_uri}/snapshots/publish_request'
        logging.info(f"Requesting publish url from {pub_url}")

        res = requests.post(url=pub_url,
                            json=publish_data,
                            headers={'Content-Type': 'application/json'})

        result = res.json()

        upload_params = result['upload_params']

        logging.info(f"Upload URL: {upload_params}")

        with open(snapshot_archive_path, 'rb') as f:
            files = {'file': ("snapshot.tar.gz",  f.read())}

        try:
            logging.info("Publish Start")
            res = requests.post(url=f"{upload_params['url']}",
                                data=upload_params['fields'],
                                files=files)
            logging.info("Publish Complete")

        except Exception as e:
            logging.error("Upload failed")
            logging.error(e)
            raise e

        return res

    def get_online_user_details(self, user_id=None):
        if user_id is None:
            user_id = self.get_user_id()
        logging.info(f"API_URL: {self.cloud_api_uri}")
        res = requests.get(url=f'{self.cloud_api_uri}/users/details',
                           params={'user_id': user_id})
        logging.info(res.content)
        return res.json()

    def get_online_agent_details(self, user_id, agent_name):
        logging.info(f"API_URL: {self.cloud_api_uri}")
        res = requests.get(url=f'{self.cloud_api_uri}/agents/details',
                           params={'user_id': user_id, "agent_name": agent_name})
        logging.info(res.content)
        return res.json()

    def get_online_agents_list(self, lookup=None):
        logging.info(f"API_URL: {self.cloud_api_uri}")
        res = requests.get(url=f'{self.cloud_api_uri}/agents/list', params={
                           'lookup': lookup, "pe": 'user_id,agent_name,created,updated'})
        return res.json().get('results', [])

    def get_online_users_list(self, lookup=None):
        logging.info(f"API_URL: {self.cloud_api_uri}")
        res = requests.get(url=f'{self.cloud_api_uri}/users/list',
                           params={'lookup': lookup, "pe": 'user_id,created'})
        return res.json().get('results', [])

    def list_published_agent_snapshots(self, agent_id, pe="snapshot_id,snapshot_data"):
        logging.info(f"API_URL: {self.cloud_api_uri}")
        res = requests.get(url=f'{self.cloud_api_uri}/snapshots/list/',
                           params={'user_id': self.get_user_id(), 'query_key': 'agent_id', "query_value": agent_id, "pe": pe})
        logging.info(res.content)

        snapshots = res.json().get('results')
        if "snapshot_data" in pe:
            for s in snapshots:
                s['snapshot_data'] = json.loads(s['snapshot_data'])
        return snapshots

    def list_published_user_snapshots(self, user_id=None, pe="snapshot_id,snapshot_data"):
        logging.info(f"API_URL: {self.cloud_api_uri}")
        if user_id is None:
            user_id = self.get_user_id()
        res = requests.get(url=f'{self.cloud_api_uri}/snapshots/list/',
                           params={'query_key': 'user_id', "query_value": user_id, "pe": pe})
        snapshots = res.json().get('results')
        if "snapshot_data" in pe:
            for s in snapshots:
                s['snapshot_data'] = json.loads(s['snapshot_data'])
        return snapshots

    def list_published_spec_snapshots(self, spec_id, pe="snapshot_id,snapshot_data"):
        logging.info(f"API_URL: {self.cloud_api_uri}")
        res = requests.get(url=f'{self.cloud_api_uri}/snapshots/list/',
                           params={'query_key': 'spec_id', "query_value": spec_id, "pe": pe})
        self.get_logger().info(res.json())
        snapshots = res.json().get('results')
        if "snapshot_data" in pe:
            for s in snapshots:
                s['snapshot_data'] = json.loads(s['snapshot_data'])
        return snapshots

    def create_agent_snapshot(
            self,
            agent_id,
            snapshot_description="",
            snapshot_version="0",
            note_addition=None):

        # Publish locally
        entity_type = 'agent'
        src_entity_path = self.get_store().get_path_from_key((entity_type, agent_id))
        dbmodel = self.get_agent_dbmodel(agent_id)

        spec_id = dbmodel.spec_id
        meta = dbmodel.meta
        notes = dbmodel.notes
        seed = dbmodel.seed
        agent_name = dbmodel.name
        config = dbmodel.config
        last_checkpoint = dbmodel.last_checkpoint
        current_timestamp = datetime.datetime.now()

        if last_checkpoint is None:
            raise Exception("Unable to create Snapshot: No checkpoints found.")

        session_data = []
        for s in dbmodel.sessions:
            session_data.append({
                'env_spec_id': s.env_spec_id,
                'env_spec_version': s.env_spec_version,
                'env_spec_config': s.env_spec_config,
                'summary': s.summary})

        if note_addition is not None:
            notes = f"[{note_addition}]\n{notes}"

        # Env Summary Stats
        # TODO: do this somewhere else, should be updated for the agent
        env_stats = {}
        def blank_stats(): return {'session_count': 0, 'step_count': 0, 'episode_count': 0,
                                   'best_ep_reward_total': None, 'best_ep_reward_mean_windowed': None}
        for s in session_data:
            stats = env_stats.get(s['env_spec_id'], blank_stats())
            # TODO: Would be better to have stats for each version and hashs for configs
            versions = stats.get('env_spec_versions', [])
            versions.append(s['env_spec_version'])
            stats['env_spec_versions'] = list(set(versions))

            if s['summary'] is not None and s['summary']['episode_count'] > 0:
                stats['step_count'] += s['summary']['step_count']
                stats['episode_count'] += s['summary']['episode_count']
                stats['session_count'] += 1
                stats['episode_count'] += s['summary']['episode_count']
                if stats['best_ep_reward_total'] is None or stats['best_ep_reward_total'] < s['summary']['best_ep_reward_total']:
                    stats['best_ep_reward_total'] = s['summary']['best_ep_reward_total']
                if stats['best_ep_reward_mean_windowed'] is None or stats['best_ep_reward_mean_windowed'] < s['summary']['best_ep_reward_mean_windowed']:
                    stats['best_ep_reward_mean_windowed'] = s['summary']['best_ep_reward_mean_windowed']
                env_stats[s['env_spec_id']] = stats

        snapshot_id = "{}_{}_{}_{}".format(spec_id,
                                           agent_id,
                                           seed,
                                           snapshot_version)
        snapshot_data = {
            'id': agent_id,
            'seed': seed,
            'agent_name': agent_name,
            'entity_type': entity_type,
            'spec_id': spec_id,
            'submitter_id': self.get_user_id(),
            'creation_time': str(current_timestamp),
            'meta': meta,
            'notes': notes,
            'last_checkpoint': last_checkpoint,
            'snapshot_description': snapshot_description,
            'snapshot_version': snapshot_version,
            'session_data': session_data,
            'env_stats': env_stats,
            'config': config,
            'snapshot_id': snapshot_id}

        temp_dir = tempfile.mkdtemp()

        # Add Config
        with open(os.path.join(temp_dir, "snapshot.json"), "w") as f:
            json.dump(snapshot_data, f, indent=2)

        # Copy Checkpoint Data to dir
        checkpoints_subdir = "checkpoints"
        last_checkpoint_id = last_checkpoint['id']
        src_checkpoints_dir = os.path.join(
            src_entity_path, checkpoints_subdir, last_checkpoint_id)
        target_checkpoints_dir = os.path.join(
            temp_dir, checkpoints_subdir, last_checkpoint_id)
        shutil.copytree(src_checkpoints_dir, target_checkpoints_dir)

        # Push to target location
        snapshot_path = os.path.join(
            self.config.local_snapshot_path, entity_type, spec_id)
        os.makedirs(snapshot_path, exist_ok=True)

        # Save Data Separately as well
        snapshot_prefix = "{}__{}_v{}".format(
            agent_id, seed, snapshot_version)
        with open(os.path.join(snapshot_path, f"{snapshot_prefix}.json"), 'w') as f:
            json.dump(snapshot_data, f, indent=2)

        # Save Data
        snapshot_filepath = os.path.join(
            snapshot_path, f"{snapshot_prefix}.tar.gz")
        make_tarfile(snapshot_filepath, temp_dir)
        self.update_snapshot_index(True)
        return snapshot_data

    def download_snapshot(self, snapshot_id):
        self.get_logger().info(
            f"Attempting to download {snapshot_id} from remote server.")

        def download_file(url, local_filepath):
            local_filename = local_filepath
            # NOTE the stream=True parameter below
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(local_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        # If you have chunk encoded response uncomment if
                        # and set chunk_size parameter to None.
                        # if chunk:
                        f.write(chunk)
            return local_filename
        res = requests.get(url=f'{self.cloud_api_uri}/snapshot/download_request/',
                           params={'user_id': self.get_user_id(), "snapshot_id": snapshot_id})
        print(res)
        self.get_logger().info(f" Server status code: {res.status_code}")
        result = res.json()
        download_url = result.get('download_url')
        snapshot_data = result.get("item").get("snapshot_data")
        dirpath = tempfile.mkdtemp()
        snapshot_archive_path = os.path.join(dirpath, "snapshot.tar.gz")

        if download_url is not None:
            download_file(download_url, snapshot_archive_path)

        return snapshot_data, snapshot_archive_path

    def create_agent_from_snapshot(self, snapshot_id):
        from .agent import Agent
        snapshot_index = self.get_snapshot_index()
        snapshot_data = snapshot_index['entries'].get(snapshot_id, None)
        if snapshot_data is None:
            self.get_logger().info("Snapshot not found localy.")
            snapshot_data, snapshot_archive_path = self.download_snapshot(
                snapshot_id=snapshot_id)
        else:
            snapshot_archive_path = "{}.tar.gz".format(os.path.join(
                self.config.local_snapshot_path, snapshot_data['path'], snapshot_data['file_prefix']))
        if not os.path.exists(snapshot_archive_path):
            src_name = snapshot_data['src']
            if src_name == "local":
                raise FileNotFoundError(
                    "Error: Snapshot Archive not found {}".format(snapshot_archive_path))
            src_name = snapshot_data['src']
            repo_url = snapshot_index['sources'][src_name]

            remote_path = os.path.join(
                repo_url, snapshot_data['path'], snapshot_data['file_prefix'])
            remote_archive_file_path = f"{remote_path}.tar.gz"
            self.get_logger().info(
                f"Snapshot not found in cache, downloading snapshot from {remote_path}")
            r = requests.get(remote_archive_file_path, allow_redirects=True)
            os.makedirs(os.path.join(self.config.local_snapshot_path,
                                     snapshot_data['path']), exist_ok=True)
            open(snapshot_archive_path, 'wb').write(r.content)
            self.get_logger().info(
                f"Snapshot downloaded to: {snapshot_archive_path}")

        temp_dir = tempfile.mkdtemp()
        extract_tarfile(snapshot_archive_path, temp_dir)
        data_source_path = os.path.join(temp_dir, 'data')

        with open(os.path.join(data_source_path, "snapshot.json"), 'r') as f:
            snapshot_data = json.load(f)

        checkpoints_src_path = os.path.join(data_source_path, "checkpoints")
        spec_id = snapshot_data['spec_id']
        meta = snapshot_data['meta']
        last_checkpoint = snapshot_data['last_checkpoint']
        meta['source_snapshot'] = copy.deepcopy(snapshot_data)
        config = snapshot_data['config']
        notes = snapshot_data['notes']
        seed = snapshot_data['seed']
        agent_name = "Clone of {}".format(
            snapshot_data['agent_name'] or snapshot_data['id'])

        agent: Agent = Agent.create(spec_id=spec_id, config=config)
        target_path = self.get_store().get_path_from_key(('agent', agent.get_id()))
        dbmodel = agent.get_dbmodel()
        dbmodel.meta = meta
        dbmodel.notes = notes
        dbmodel.seed = seed
        dbmodel.name = agent_name
        dbmodel.last_checkpoint = last_checkpoint
        import shutil
        shutil.copytree(checkpoints_src_path, os.path.join(
            target_path, "checkpoints"))
        self.get_dbsession().commit()
        return agent

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
                info['gpu_list'] = {i: torch.cuda.get_device_name(
                    0) for i in range(gpu_count)}
                dev_id = torch.cuda.current_device()
                info['current_gpu_device_id'] = dev_id
                info['current_gpu_device_name'] = torch.cuda.get_device_name(
                    dev_id)
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
        info['gpu_list'] = {
            x.incarnation: x.name for x in local_device_protos if x.device_type == 'GPU'}
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
