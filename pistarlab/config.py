import logging
import os
from pathlib import Path
from pistarlab.util_funcs import merged_dict
from typing import Any, Dict

import yaml
from pistarlab.meta import DEFAULT_REDIS_PASSWORD

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
            log_level,
            db_type,
            db_config,
            redis_hostname,
            redis_port,
            redis_password,
            execution_context_config,
            enable_cluster,
            streamer_uri=None,
            read_only_mode=False,
            enable_ide=False,
            extension_config={}):

        self.root_path: str = root_path
        self.read_only_mode = read_only_mode

        self.data_path = os.path.join(root_path, "data")
        os.makedirs(self.data_path, exist_ok=True)

        self.log_root = os.path.join(root_path, "logs")
        os.makedirs(self.log_root, exist_ok=True)

        self.workspace_path = os.path.join(root_path, 'workspace')
        os.makedirs(self.workspace_path, exist_ok=True)

        self.enable_ide = enable_ide
        self.enable_cluster = enable_cluster

        self.local_snapshot_path = os.path.join(
            self.root_path, "snapshot_repo")
        self.snapshot_index_path = os.path.join(
            self.root_path, 'snapshot_index.json')
        os.makedirs(self.local_snapshot_path, exist_ok=True)
        self.streamer_uri = streamer_uri

        self.log_level: str = log_level
        self.db_type = db_type
        self.db_config = db_config

        self.redis_hostname = redis_hostname
        self.redis_port = redis_port
        self.redis_password = redis_password
        self.extension_config = extension_config

        self.execution_context_config: Dict[str,
                                            Any] = execution_context_config
        if self.enable_cluster:
            with open(os.path.join(self.root_path, 'cluster.yaml'), 'r') as f:
                self.cluster_config = yaml.load(f, Loader=yaml.FullLoader)
            self.execution_context_config['address'] = self.cluster_config['provider']['head']
        else:
            self.cluster_config = None

        if self.db_type == "sqlite":
            self.db_connection_string: str = "sqlite:///{}/data.db".format(
                self.data_path)
        else:
            db_hostname = db_config.get("db_hostname")
            if db_hostname is None or db_hostname == "":
                raise Exception(
                    f"db_hostname is required but was not provided in db_config")
            self.db_connection_string: str = f'postgresql+psycopg2://{db_config.get("db_user","postgres")}:{db_config.get("db_password","pistarlab")}@{db_hostname}/{db_config.get("db_name","postgres")}'


def load_sys_config_file(root_path):
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
        "redis_password": DEFAULT_REDIS_PASSWORD,
        "enable_ide": False,
        "extension_config": {
            "install_on_boot": [{
                'id': 'pistarlab-envs-gym-main',
                'version': "0.0.1-dev"},
                {
                'id': 'pistarlab-rllib',
                'version': "0.0.1-dev"},
                {
                'id': 'pistarlab-landia',
                'version': "0.0.1-dev"}
            ]
        }
    }

    config_path = os.path.join(root_path, "config.yaml")
    if os.path.isfile(config_path):
        with open(config_path, 'r') as f:
            config.update(yaml.full_load(f))
    else:
        logging.info(
            "No Config File Found at {}. Creating a config file with default values.".format(config_path))
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
    return config


def update_config(root_path, config_overrides):
    config = load_sys_config_file(root_path)
    updated_config = merged_dict(config, config_overrides)
    config_path = os.path.join(root_path, "config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(updated_config, f)
    # TODO: avoid restart
    logging.info("Config updated, Please restart for changes to take effect")


def get_sys_config(root_path=None):
    if root_path is None:
        root_path = get_root_path()

    os.makedirs(root_path,exist_ok=True)
    config = load_sys_config_file(root_path)

    return SysConfig(
        root_path=root_path,
        **config)
