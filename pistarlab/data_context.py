
import json
import logging
import os
import shutil
import time

import redis
import sqlalchemy as db
from sqlalchemy import orm as db_orm

from .config import SysConfig
from .dbmodels import DbBase
from .storage import JSONEncoderCustom, Storage
from .utils.misc import gen_shortuid


def json_data_serializer(d):
    return json.dumps(d, cls=JSONEncoderCustom)


def get_user_info(config: SysConfig):
    user_info_path = os.path.join(config.root_path, "user_info.json")
    if not os.path.exists(user_info_path):
        with open(user_info_path,'w') as f:
            json.dump({'user_id':gen_shortuid()},f)

    try:
        with open(user_info_path, "r") as f:
            return json.load(f)

    except Exception as e:
        logging.error(f"ERROR loading display info {e}")
        return {}

class DataContext:

    def __init__(self, config: SysConfig):
        self.config = config

        # Database
        self._dbengine = None
        self._dbsession = None
        self.__db_scoped_session = None
        self.redis_client: redis.Redis = None
        self.user_info = None

        # File Storage
        self._store: Storage = None

    def init_db(self, force=False):
        engine = self.get_dbengine(reset=True)
        if not engine.dialect.has_table(engine, 'agent') or force:
            logging.info("Created database")
            DbBase.metadata.create_all(engine)
            from .dbinit import load_default_data
            load_default_data()

    def get_redis_client(self):
        if self.redis_client is None:
            self.redis_client = redis.Redis(
                host=self.config.redis_hostname,
                port=self.config.redis_port,
                password=self.config.redis_password)
        return self.redis_client

    def cleanup(self):
        try:
            logging.debug("Updating statuses, if needed")
            for entity_name in ['session', 'task']:
                self.get_dbsession().execute(
                    """UPDATE {} SET status = 'ABORTED', status_msg = 'Running during prior instance and not shutdown properly'
                    where status in ('RUNNING','STARTED')""".format(entity_name))

            self.get_dbsession().commit()
        except Exception as e:
            self.get_dbsession().rollback()
            logging.error(e)

    def _get_db_scoped_session(self) -> db_orm.scoped_session:
        if self.__db_scoped_session is None:
            self.__db_scoped_session = db_orm.scoped_session(db_orm.sessionmaker(self.get_dbengine(), autoflush=True, autocommit=False))
        return self.__db_scoped_session

    def _drop_db(self, are_you_sure=False):
        if are_you_sure:
            self.get_dbsession().rollback()
            logging.info("Dropping database")
            DbBase.metadata.drop_all(self.get_dbengine())
            # from sqlalchemy_utils.functions import drop_database

            # drop_database(self.get_dbengine().url)
            logging.info("database dropped")

    # PUBLIC METHODS
    def get_dbsession(self) -> db_orm.Session:
        if self._dbsession is None:
            self._dbsession = self._get_db_scoped_session()
        return self._dbsession()

    def get_store(self) -> Storage:
        if self._store is None:
            self._store: Storage = Storage(self.config.data_path, use_write_thread=True)
        return self._store

    def get_dbengine(self, reset=False):
        if reset and self._dbengine is not None:
            try:
                self._dbengine.dispose()
            except:
                pass
            self._dbengine = None
        if self._dbengine is None:
            self._dbengine = db.create_engine(
                self.config.db_connection_string,
                convert_unicode=True,
                echo=False,
                json_serializer=json_data_serializer)
        return self._dbengine

    def reset_all_data(self):
        logging.info("Resetting Core Data")
        source_dir_path = self.config.data_path
        backup_id = "{}".format(time.strftime("%Y%m%d-%H%M%S"))
        if os.path.exists(source_dir_path):
            target_dir_path = os.path.join(self.get_store().root_path + "_backup_{}".format(backup_id))
            if os.path.exists(target_dir_path):
                logging.info("Removing old backup:{}".format(target_dir_path))
                shutil.rmtree(target_dir_path)
            shutil.move(source_dir_path, target_dir_path)

        try:
            self._drop_db(are_you_sure=True)
        except Exception as e:
            logging.error("Unable to drop database {}".format(e))
        finally:

            self.init_db()

    def get_user_id(self, token= None):
        # TODO: need to replace this...
        if self.user_info is None:
            self.user_info = get_user_info(self.config)
        return self.user_info.get("user_id") 

    def close(self):
        if self.__db_scoped_session:
            self.__db_scoped_session.remove()
        if self._store is not None:
            self._store.close()
