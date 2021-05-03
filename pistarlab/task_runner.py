import time
import traceback
from abc import ABCMeta
from typing import Any, Dict, List

from . import ctx
from .meta import *
from .task import AgentTask, Task
from .util_funcs import *
from .util_funcs import get_class_from_entry_point
import sys


class TaskRunner(metaclass=ABCMeta):

    description = None
    spec_id = None
    displayed_name = None
    entry_point = 'pistarlab.task:Task'
    plugin_id = "WORKSPACE"
    plugin_version = "0.0.1-dev"
    version = "0.0.1-dev"
    config = {}


    def __init__(self, task_id):
        self._logger = None
        self.task_id = task_id

        ctx.connect(task_id)
        ctx.set_default_logger(self.get_logger())
        self.get_logger().info("Task Runner Started")

        self.task: Task = Task.load(self.task_id)
        self.task.change_status(STATE_RUNNING)
        self.last_check = 0
        task_config = self.get_task().get_config()
        self.status_check_freq_secs = task_config.get('status_check_freq_secs', 4)

    def get_logger(self):
        if self._logger is None:
            self._logger = ctx.get_entity_logger(TASK_ENTITY, self.task_id,sub_id="runner")
        return self._logger

    def initialize(self):
        pass

    def close(self):
        ctx.close()
        return True

    def get_task(self) -> Task:
        return self.task

    def is_running(self):  # Check for terminination request
        running = True
        if (time.time() - self.last_check > self.status_check_freq_secs):
            if self.task.get_status() != STATE_RUNNING:
                self.get_logger().info("Terminating work for {} as requested".format(self.task_id))
                running = False
            self.last_check = time.time()
        return running

    def _run(self):
        task = self.get_task()
        exit_state = STATE_ABORTED
        exit_msg = ""
        try:
            try:
                summary, exit_state = self.run()
            except:
                e = sys.exc_info()[0]
                msg = "Exception while executing task runner {}. traceback {}".format(e, traceback.format_exc())
                self.get_logger().error(msg)
                exit_state = STATE_ABORTED
        # Will throw exception, this is expected and handled by the RunFuture object
        except Exception as e:
            exit_state = STATE_ABORTED
        finally:
            task.shutdown(exit_state, exit_msg)

    def run(self) -> Tuple[Any, Any]:
        raise NotImplementedError("No run method implemented for task runner: {}".format(type(self).__name__))


class AgentTaskRunner(TaskRunner):

    # def __init__(self,*args,**kwargs):
    #     super().__init__(*args,**kwargs)

    def get_task(self) -> AgentTask:
        return super().get_task()
