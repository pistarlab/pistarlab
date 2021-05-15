import logging
import os
from pathlib import Path
from typing import Any, Dict

import yaml

DEFAULT_EXECUTION_CONTEXT_CONFIG = {
    'address': 'auto',
    'ignore_reinit_error': True,
    'configure_logging': False}


def get_root_path():
    return os.getenv("PISTARLAB_ROOT", os.path.join(str(Path.home()), "pistarlab"))


def get_ip():
    import socket
    return socket.gethostbyname(socket.gethostname())


class SysConfig:

    def __init__(
            self,
            root_path,
            data_path,
            workspace_path,
            log_root,
            log_level,
            db_type,
            db_config,
            redis_hostname,
            redis_port,
            redis_password,
            execution_context_config,
            enable_cluster,
            streamer_uri=None,
            read_only_mode=False):

        self.root_path: str = root_path
        self.data_path: str = data_path
        self.read_only_mode = read_only_mode

        self.workspace_path: str = workspace_path
        self.log_root = log_root

        self.enable_cluster = enable_cluster

        self.local_snapshot_path = os.path.join(self.root_path, "snapshot_repo")
        self.snapshot_index_path = os.path.join(self.root_path, 'snapshot_index.json')
        os.makedirs(self.local_snapshot_path, exist_ok=True)
        self.streamer_uri = streamer_uri

        self.log_level: str = log_level
        self.db_type = db_type
        self.db_config = db_config

        self.redis_hostname = redis_hostname
        self.redis_port = redis_port
        self.redis_password = redis_password

        self.execution_context_config: Dict[str, Any] = execution_context_config
        if self.enable_cluster:
            with open(os.path.join(self.root_path, 'cluster.yaml'), 'r') as f:
                self.cluster_config = yaml.load(f, Loader=yaml.FullLoader)
            self.execution_context_config['address'] = self.cluster_config['provider']['head']
        else:
            self.cluster_config = None

        if self.db_type == "sqlite":
            self.db_connection_string: str = "sqlite:///{}/data.db".format(data_path)
        else:
            db_hostname = db_config.get("db_hostname")
            if db_hostname is None or db_hostname == "":
                raise Exception(f"db_hostname is required but was not provided in db_config")
            self.db_connection_string: str = f'postgresql+psycopg2://{db_config.get("db_user","postgres")}:{db_config.get("db_password","pistarlab")}@{db_hostname}/{db_config.get("db_name","postgres")}'


def get_sys_config(root_path=None):
    if root_path is None:
        root_path = get_root_path()

    data_path = os.path.join(root_path, "data")
    log_root = os.path.join(root_path, "logs")
    workspace_path = os.path.join(root_path, 'workspace')

    config = {
        'db_type': "sqlite",
        'db_config': {'db_user': None, 'db_password': None, 'db_hostname': None, 'db_name': None},
        'execution_context_config': DEFAULT_EXECUTION_CONTEXT_CONFIG,
        'log_level': "INFO",
        'enable_cluster': False,
        'streamer_uri': f"http://localhost:7778/offer",
        "read_only_mode": False,
        "redis_hostname": "localhost",
        "redis_port": "7771",
        "redis_password": "5241590000000000",
    }

    os.makedirs(data_path, exist_ok=True)
    os.makedirs(workspace_path, exist_ok=True)
    os.makedirs(log_root, exist_ok=True)

    config_path = os.path.join(root_path, "config.yaml")
    if os.path.isfile(config_path):
        with open(config_path, 'r') as f:
            config.update(yaml.full_load(f))
    else:
        logging.info("No Config File Found at {}. Creating a config file with default values.".format(config_path))
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

    return SysConfig(
        root_path=root_path,
        data_path=data_path,
        workspace_path=workspace_path,
        log_root=log_root,
        **config)
