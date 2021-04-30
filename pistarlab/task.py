import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, List

from . import ctx
from .agent import Agent
from .dbmodels import TaskModel, TaskSpecModel
from .entity import Entity
from .execution_context import RunFuture
from .meta import *
from .util_funcs import *
from .util_funcs import get_class_from_entry_point
from .utils.misc import gen_shortuid


class Task(Entity):

    @staticmethod
    def load(id):
        dbmodel: TaskModel = Task.get_dbmodel_by_id(id)
        cls = get_class_from_entry_point(dbmodel.spec.entry_point)
        return cls(
            _id=dbmodel.id,
            spec_id=dbmodel.spec.id,
            config=dbmodel.config)

    @staticmethod
    def create(spec_id, config, parent_task_id=None):
        spec_dbmodel: TaskSpecModel = Task.get_spec_dbmodel_by_id(spec_id)
        cls = get_class_from_entry_point(spec_dbmodel.entry_point)
        config = merged_dict(spec_dbmodel.config, config)
        return cls(
            spec_id=spec_id,
            config=config,
            parent_task_id=parent_task_id)

    @staticmethod
    def get_dbmodel_by_id(id) -> TaskModel:
        query = ctx.get_dbsession().query(TaskModel)
        return query.get(id)

    @staticmethod
    def get_spec_dbmodel_by_id(id) -> TaskSpecModel:
        query = ctx.get_dbsession().query(TaskSpecModel)
        return query.get(id)

    def set_primary_session_by_id(self, session_id):
        dbmodel = ctx.get_dbsession()
        model = self.get_dbmodel()
        model.primary_session_id = session_id
        dbmodel.commit()

    def __init__(self, _id=None, spec_id=None, config={}, parent_task_id=None, archive=False):
        super().__init__(_id=_id, entity_type=TASK_ENTITY)
        self._config = config
        self.parent_task_id = parent_task_id
        self.spec_id = spec_id
        self.archive = archive

        self._sync_data_model()
        self.summary_sub = None
        self.runner = None
        self.run_future = None
        self._logger = None

    def update_config(self, config):
        if self.get_status() != STATE_CREATED:
            raise Exception("Unable to update configuration after task has been run")
        self._config = merged_dict(self._config, config)
        self.get_dbmodel().config = self._config
        ctx.get_dbsession().commit()

    def get_dbmodel(self) -> TaskModel:
        db_model = Task.get_dbmodel_by_id(self._id)
        if db_model is None:
            db_model = self._sync_data_model()
        return db_model

    def _sync_data_model(self):
        dbsession = ctx.get_dbsession()
        dbmodel = Task.get_dbmodel_by_id(self._id)
        if dbmodel is None:
            logging.info(f"{self.entity_type} with id:{self._id} not found, creating new record.")
            dbmodel = TaskModel(
                id=self._id,
                spec_id=self.spec_id,
                parent_task_id=self.parent_task_id,
                config=self._config,
                archived=self.archive)
            dbsession.add(dbmodel)
            dbsession.commit()
        return dbmodel

    def change_status(self, state, msg=''):
        try:
            dbmodel = self.get_dbmodel()
            dbmodel.status = state
            dbmodel.status_msg = msg
            dbmodel.status_timestamp = datetime.now()
            ctx.get_dbsession().commit()
        except Exception as e:
            ctx.get_dbsession().rollback()
            raise e
        return True

    def get_status(self):
        return self.get_dbmodel().status

    def update_summary(self, summary):
        summary = clean_collection(summary)
        ctx.get_store().save((TASK_ENTITY, self.get_id()), name="summary", value=summary)
        dbmodel = self.get_dbmodel()
        dbmodel.summary = summary
        ctx.get_dbsession().commit()

    def wait_for_next_summary_update(self):
        if self.summary_sub is None:
            self.summary_sub = ctx.get_redis_client().pubsub()
            self.summary_sub.subscribe("TASK_SUMMARY_STATS_{}".format(self.get_id()))
        msg = next(self.summary_sub.listen())
        data = 1
        if msg is not None:
            channel = msg['channel']
            data = msg['data']
        if data == 1 or data == 2 or data == None:
            results = {}
        else:
            results = json.loads(data.decode('utf-8'))
        return results

    def get_summary(self):
        return self.get_dbmodel().summary

    def set_actor_uid(self, actor_uid):
        dbmodel = self.get_dbmodel()
        dbmodel.actor_uid = actor_uid
        ctx.get_dbsession().commit()

    def get_actor_uid(self) -> str:
        return self.get_dbmodel().actor_uid

    def get_child_tasks(self) -> List[Any]:
        tasks = []
        for subtask in self.get_dbmodel().child_tasks:
            tasks.append(Task.load(subtask.id))
        return tasks

    def get_child_task_summaries(self):
        task_model = self.get_dbmodel()
        summaries = []
        for subtask in task_model.child_tasks:
            summaries.append(subtask.summary)
        return summaries

    def get_logger(self):
        if self._logger is None:
            self._logger = ctx.get_entity_logger(TASK_ENTITY, self._id, sub_id="task")
        return self._logger

    def shutdown(self, shutdown_state=STATE_TERMINATED, msg=""):
        """
        TODO: What is the difference between this and _shutdown?
        """
        from .session import Session
        sess = self.get_dbmodel().primary_session
        if sess is not None and sess.status == STATE_RUNNING:
            Session.change_status_by_id(sess.id, STATE_TERMINATED)

        if self.get_status() != STATE_RUNNING:
            return
        logging.info("Shutting down task runner{}".format(self.get_id()))
        child_tasks = self.get_child_tasks()
        for child_task in child_tasks:
            child_task._shutdown(shutdown_state=STATE_TERMINATED)
        self.change_status(state=shutdown_state, msg=msg)

    def _shutdown(self, shutdown_state=STATE_TERMINATED, msg=""):
        if self.get_status() != STATE_RUNNING:
            return
        try:
            logging.info("Shutting down task runner{}".format(self.get_id()))
            child_tasks = self.get_child_tasks()
            for child_task in child_tasks:
                child_task._shutdown(shutdown_state=STATE_TERMINATED)
            ctx.get_execution_context().get(self._get_runner().close.remote())
        finally:
            self.change_status(state=shutdown_state, msg=msg)
            logging.info("Stopping task actor now {}".format(self.get_actor_uid()))
            ctx.get_execution_context().stop_actor_by_uid(self.get_actor_uid())

    def _get_runner(self):
        if self.runner is None:
            try:
                self.runner = ctx.get_execution_context().get_actor_by_uid(self.get_dbmodel().actor_uid)
            except:
                try:
                    actor_uid = gen_shortuid()
                    self.set_actor_uid(actor_uid)
                    cls = ctx.get_execution_context().as_actor(get_class_from_entry_point(self.get_dbmodel().spec.runner_entry_point))
                    self.runner = cls.options(name=actor_uid, max_concurrency=10).remote(task_id=self.get_id())
                except:
                    e = sys.exc_info()[0]
                    self.get_logger().info(f"Error starting task (id:{self._id}) runner {e}")
                    self.shutdown()
                    raise e
        return self.runner

    def run(self) -> RunFuture:
        # TODO: add local mode support
        if self.get_status() == STATE_RUNNING:
            logging.error("Task is already running")
        runner = self._get_runner()

        try:
            ray_task_id = runner._run.remote()
            self.run_future = RunFuture(ray_task_id, callback=lambda: self.get_summary())
        except Exception as e:
            self.change_status(STATE_ABORTED, "Task failed when running: {}".format(e))
            raise e

        return self.run_future


class AgentTask(Task):

    @staticmethod
    def create(
            env_spec_id=None,
            env_kwargs={},
            parent_task_id=None,
            session_config={},
            agent: Agent = None,
            agent_run_config={},
            batch_size=1,
            status_check_freq_secs=2,
            checkpoint_freq=10,
            archive=False,
            interface_id="run",
            **kwargs):

        config = {}
        if not type(session_config) is dict:
            session_config = session_config.get_dict()

        config['session_config'] = session_config
        config['env_spec_id'] = env_spec_id
        config['env_kwargs'] = env_kwargs
        config['agent_id'] = agent.get_id()
        config['agent_run_config'] = agent_run_config
        config['batch_size'] = batch_size
        config['checkpoint_freq'] = checkpoint_freq
        config['status_check_freq_secs'] = status_check_freq_secs
        config['interface_id'] = interface_id
        config.update(kwargs)

        return AgentTask(
            config=config,
            spec_id="agent_task",
            parent_task_id=parent_task_id,
            archive=archive)

    def update_summary(self, summary):
        super().update_summary(summary)
        # ctx.get_redis_client().publish(
        #     "TASK_SUMMARY_STATS_{}".format(self.get_id()),
        #     json.dumps(self.get_session_summary_stats()))

    def get_agent_dbmodel(self):
        return ctx.get_agent_dbmodel(self.get_dbmodel().config['agent_id'])

    def get_agent(self) -> Agent:
        return Agent.load(self.get_dbmodel().config['agent_id'])

    def get_agent_run_config(self):
        return self.get_config()['agent_run_config']

    def get_session_summaries(self):
        task_model = self.get_dbmodel()
        summaries = []
        for sess in task_model.sessions:
            summaries.append(sess.summary)
        return summaries

    def get_sessions(self):
        task_model = self.get_dbmodel()
        ids = []
        for sess in task_model.sessions:
            ids.append(sess.id)
        return ids

    def _get_runner(self):
        if self.runner is None:
            try:
                self.runner = ctx.get_execution_context().get_actor_by_uid(self.get_dbmodel().actor_uid)
            except Exception as e:
                try:
                    actor_uid = gen_shortuid()
                    self.set_actor_uid(actor_uid)
                    # Use agent runner
                    runner_entry_point = self.get_agent_dbmodel().spec.runner_entry_point
                    cls = ctx.get_execution_context().as_actor(get_class_from_entry_point(runner_entry_point))
                    self.runner = cls.options(name=actor_uid, max_concurrency=10).remote(task_id=self.get_id())
                except Exception as e:
                    self.get_logger().info("Error creating runner for task id:{}. Error: {}".format(self._id, e))
                    self.shutdown(STATE_ABORTED, exit_msg="FAILED TO CREATE TASK RUNNER")
                    raise e
        return self.runner

    def shutdown(self,exit_state=STATE_TERMINATED, exit_msg="" ):
        try:
            from .session import Session
            for sess in self.get_dbmodel().sessions:
                if sess.status == STATE_RUNNING:
                    Session.change_status_by_id(sess.id, STATE_TERMINATED)
        finally:
            super().shutdown(exit_state, exit_msg)

    def get_session_summary_stats(self):
        return get_stats_flat(transpose_dict_list(self.get_session_summaries()), suffix='task')

    #TODO: Not sure if we want to keep this
    # def session_env_instance(self):
    #     task_id = self.get_id()
    #     task_config = self.get_config()

    #     agent = self.get_agent()

    #     env_spec_id = task_config['env_spec_id']
    #     env_kwargs = task_config['env_kwargs']

    #     session_config = task_config['session_config']
    #     agent_run_config = agent.get_config(task_config['agent_run_config'])

    #     batch_size = task_config.get('batch_size',1)
    #     use_remote_client = task_config.get('use_remote_client',False)

    #     from .session_env import RLMultiSessionEnv
    #     from .session_config import RLSessionConfig
    #     return RLMultiSessionEnv(
    #             env_spec_id=env_spec_id,
    #             env_kwargs=env_kwargs,
    #             config=RLSessionConfig(**session_config),
    #             agent_id=agent.get_id(),
    #             agent_run_config=agent_run_config,
    #             batch_size = batch_size,
    #             task_id=task_id,
    #             use_remote_client=use_remote_client)
