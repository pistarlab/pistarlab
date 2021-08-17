import json
import logging
import struct
import time
import traceback
from abc import abstractmethod
from datetime import datetime
from typing import Any, Dict, Tuple

from .meta import *

from . import ctx
from .databuffer import DataBuffer
from .dbmodels import EnvSpecModel, SessionModel, TaskModel
from .entity import Entity
from .meta import RL_SINGLEPLAYER_SESS, SESSION_TYPES
from .session_config import DataSessionConfig, RLSessionConfig, SessionConfig
from .util_funcs import *
from .utils.env_helpers import (render_env_frame_as_image)


class Session(Entity):

    @staticmethod
    def get_summary_by_id(session_id):
        return Session.get_dbmodel_by_id(session_id).summary

    @staticmethod
    def get_logs_by_id(session_id, name):
        return ctx.get_store().get_session_logs(session_id, name)

    @staticmethod
    def get_dbmodel_by_id(id) -> SessionModel:
        query = ctx.get_dbsession().query(SessionModel)
        return query.get(id)

    @staticmethod
    def change_status_by_id(id, state, msg=''):
        try:
            dbmodel = Session.get_dbmodel_by_id(id)
            dbmodel.status = state
            dbmodel.status_msg = msg
            dbmodel.status_timestamp = datetime.now()
            ctx.get_dbsession().commit()
        except Exception as e:
            ctx.get_dbsession().rollback()
            raise e
        return True

    def __init__(self,
                 env_spec_id,
                 env_kwargs={},
                 config: SessionConfig = None,
                 agent_id=None,
                 agent_run_config={},
                 task_id=None,
                 session_type=None,
                 parent_session_id=None):
        _id = "TMP_SESSION" if config.tmp_session is True else None
        super().__init__(_id=_id, entity_type=SESSION_ENTITY)
        self.env_spec_id = env_spec_id
        self.env_kwargs = env_kwargs
        self.config = config
        self.running = True
        self.agent_id = agent_id
        self.agent_run_config = agent_run_config
        self.task_id = task_id
        self.session_type = session_type
        self.parent_session_id = parent_session_id
        self._logger = None

    def get_logger(self):
        if self._logger is None:
            self._logger = ctx.get_entity_logger(SESSION_ENTITY, self._id)
        return self._logger

    def get_dbmodel(self) -> SessionModel:
        return Session.get_dbmodel_by_id(self._id)

    def get_spec_dbmodel(self) -> EnvSpecModel:
        return ctx.get_dbsession().query(EnvSpecModel).get(self.env_spec_id)

    def _sync_data_model(self):
        # NOTE: Should only be called once
        dbsession = ctx.get_dbsession()
        try:
            dbmodel = self.get_dbmodel()
            if dbmodel is None:
                self.get_logger().info("Creating new Session. {}/{}".format(self._id, self.session_type))
                dbmodel = SessionModel(
                    id=self._id,
                    env_spec_id=self.env_spec_id,
                    config=self.config.get_dict(),
                    task_id=self.task_id,
                    agent_id=self.agent_id,
                    session_type=self.session_type,
                    agent_run_config=self.agent_run_config,
                    parent_session_id=self.parent_session_id)
                dbsession.add(dbmodel)
                dbsession.commit()
            else:
                logging.info("Session already created {}".format(self._id))
        except (Exception, KeyboardInterrupt) as e:
            dbsession.rollback()
            raise e

    def change_status(self, state, msg=''):
        return Session.change_status_by_id(self._id, state, msg)

    def set_parent_session_by_id(self, session_id):
        dbmodel = self.get_dbmodel()
        dbmodel.parent_session_id = session_id
        ctx.get_dbsession().commit()

    def add_child_session_by_id(self, session_id):
        child_session_model = self.get_dbmodel_by_id(session_id)
        child_session_model.parent_session_id = self.get_id()
        ctx.get_dbsession().commit()

    def set_as_primary_session_for_task_by_id(self, task_id):
        task: TaskModel = ctx.get_dbsession().query(TaskModel).get(task_id)
        task.primary_session_id = self.get_id()
        ctx.get_dbsession().commit()

    def get_status(self):
        return self.get_dbmodel().status

    def save_summary(self):
        try:
            dbmodel = self.get_dbmodel()
            if dbmodel is None:
                self.get_logger().error("DBModel is None for session {}".format(self._id))
            dbmodel.summary = self.get_summary()
            ctx.get_dbsession().commit()
        except (Exception, KeyboardInterrupt) as e:
            ctx.get_dbsession().rollback()
            raise e

    def save_run_info(self):
        try:
            dbmodel = self.get_dbmodel()
            if dbmodel is None:
                self.get_logger().error("DBModel is None for session {}".format(self._id))
                dbmodel.run_info = self.get_run_info()
            ctx.get_dbsession().commit()
        except (Exception, KeyboardInterrupt) as e:
            ctx.get_dbsession().rollback()
            raise e

    def _initialize(self):
        self._sync_data_model()
        self.session_start_time = time.time()
        self.change_status(STATE_RUNNING)
        self.save_summary()
        self.initialized = True
        self.save_run_info()
        self.get_logger().info("Session Initialized")

    def close(self, state):
        self.change_status(state)
        self.get_logger().info("Ending Session")
        self.running = False

    @abstractmethod
    def get_run_info(self):
        raise NotImplementedError()

    @abstractmethod
    def get_summary(self):
        raise NotImplementedError()

    @abstractmethod
    def is_complete(self):
        raise NotImplementedError()


class DataSession(Session):

    def __init__(self, config: DataSessionConfig = None, *args, **kwargs):
        config = config or DataSessionConfig()
        super().__init__(config=config, session_type=DATA_SESSION, *args, **kwargs)

        self.data_batch_log_flush_interval_secs = 10
        self.last_data_batch_log_flush = 0

        self.data_metrics_buffer: DataBuffer = None
        self.eval_metrics_buffer: DataBuffer = None

        self.session_start_time = time.time()

        self.data_metrics = {}
        self.eval_metrics = {}

        self._initialize()

    def get_run_info(self):
        data = {}
        return data

    def get_summary(self):
        summary = {
            'start_timestamp': self.session_start_time,
            'data_metrics': self.data_metrics,
            'eval_metrics': self.eval_metrics,
        }
        return summary

    def close(self, state=None):

        self._flush_data_log()
        self.save_summary()

        if state is None:
            if self.is_complete():
                state = STATE_COMPLETED
            else:
                state = STATE_TERMINATED

        super().close(state)
        return state

    def log_data_metrics(self, metrics):
        metrics['timestamp'] = time.time()
        if self.data_metrics_buffer is None:
            self.data_metrics_buffer = DataBuffer(
                cols=list(metrics.keys()),
                size=1000)
        self.data_metrics_buffer.add_dict(metrics)
        self._flush_data_log()

    def log_evaluation_metrics(self, metrics):
        metrics['timestamp'] = time.time()
        if self.eval_metrics_buffer is None:
            self.eval_metrics_buffer = DataBuffer(
                cols=list(metrics.keys()),
                size=1000)
        self.eval_metrics_buffer.add_dict(metrics)

        # Flush
        ep_log_data = self.eval_metrics_buffer.get_dict()
        ctx.get_store().save_session_logs(self._id, "eval_batch_stats", ep_log_data)
        self.eval_metrics_buffer.clear()

    def log_predictions(self, name, inputs, outputs, predictions, metrics):
        # TODO:
        pass

    def _flush_data_log(self):
        if self.data_metrics_buffer.counter > 0:
            ep_log_data = self.data_metrics_buffer.get_dict()
            ctx.get_store().save_session_logs(self._id, "data_batch_stats", ep_log_data)
            self.data_metrics_buffer.clear()
        self.last_data_batch_log_flush = time.time()

    def is_complete(self):
        pass


class RLSession(Session):

    def __init__(
            self,
            session_type,
            config: RLSessionConfig = None,
            run_meta={},
            env_run_meta={},
            render_fn=None,
            persists_at_creation=False,
            *args, **kwargs):
        config = config or RLSessionConfig()
        assert(session_type in SESSION_TYPES and session_type.startswith("RL_")), "INVALID SESSION TYPE {}".format(session_type)

        super().__init__(config=config, session_type=session_type, *args, **kwargs)

        self.config: RLSessionConfig = self.config  # For type inference

        self.is_group_session = self.session_type != RL_SINGLEPLAYER_SESS

        spec_model = self.get_spec_dbmodel()

        # Override
        self.env_run_meta = merged_dict(spec_model.meta, env_run_meta)
        self.render_fn = render_fn

        self.run_meta = run_meta

        self.episode_counter = 0
        self.step_counter = 0
        self.episode_step_counter = 0
        self.reset_episode = True
        self.episode_reward_total = 0
        self.reward_total = 0

        self.prev_observation = None
        self.prev_reward = None
        self.prev_done = None
        self.prev_action = None
        self.prev_info = None

        self.session_start_time = time.time()

        self.last_tick = 0
        self.step_speed_limitor = None

        self.render_mode = self.env_run_meta.get("render_mode")
        self.render_stats_enabled = self.env_run_meta.get("render_stats")

        self.recording = False
        self.recording_interval_secs = 10.0 # TODO: is this used?
        self.recording_last_timestamp = 0
        self.continuous_recording_enabled = False
        self.render_frame_enabled = self.config.preview_rendering

        self.runtime_logging = True

        # Publish render/preview frame to redis
        self.frame_stream_enabled = self.config.frame_stream_enabled
        self.last_frame_update_timestamp = 0
        self.frame_update_interval_secs = 0.1

        # User Command Check/Status Publish
        self.last_command_check = 0
        self.command_check_interval = 3
        self.last_command_update =0
        
        self.pub = ctx.get_redis_client().pubsub()
        self.pub.subscribe(f"SESSION_COMMAND_REQUEST_{self._id}")

        self.last_stats_update_timestamp = 0
        self.stats_update_interval_secs = 1

        # Experience History
        self.history = DataBuffer(
            cols=[
                'runtime',
                'step_count',
                'episode_count',
                'observation',
                'reward',
                'action',
                'done',
                'info',
                'reward_total',
                'episode_step_count',
                'episode_reward_total',
                'env_stats'],
            size=10000)

        self.log_flush_interval_secs = 1.0

        # step logs
        self.step_log_mean_latency = 0
        self.step_log_last_flush = 0
        self.step_log_last_step_count = 0
        self.step_log_last_timestamp = time.time()
        self.step_log_buffer = DataBuffer(
            cols=[
                'runtime',
                'step_count',
                'episode_count',
                'reward_total',
                'step_latency_mean_windowed'],
            size=10000)

        # episode logs
        self.ep_log_last_flush = 0
        self.ep_log_buffer = DataBuffer(
            cols=[
                'runtime',
                'step_count',
                'episode_count',
                'episode_step_count',
                'episode_reward_total'],
            size=10000)

        self.session_start_time = time.time()

        self.prev_summary = None

        # best episode tracking
        self.best_ep_reward_total = None
        self.best_ep_reward_total_ep_count = 0
        self.last_ep_reward_mean_windowed = None
        self.best_ep_reward_mean_windowed = None
        self.best_ep_reward_mean_windowed_ep_count = 0

        self.initialized = False
        if persists_at_creation:
            self._sync_data_model()

    def run_status_update(self):
        self.get_logger().debug("Issuing command check")
        reading_messages = True
        while reading_messages:
            msg = self.pub.get_message()
            if msg:
                # channel = msg['channel']
                
                if "data" not in msg:
                    self.get_logger().info(f"Unexpected message? raw message value: {msg}")
                    continue
                data = msg['data']
                if data is None or data == 1 or data == 2:
                    reading_messages = False
                else:
                    
                    command_data = json.loads(data.decode('utf-8'))
                    msg = "Received command update {}".format(command_data)
                    self.get_logger().info(msg)
                    command = command_data.get("command")
                    params = command_data.get("params")
                    self.last_command_update = time.time()
                    if command == "recording":    
                        self.continuous_recording_enabled = params.get("enabled",False)
                        self.get_logger().info(f"Set recording to {self.continuous_recording_enabled}")
                    elif command == "runtime_logging":    
                        self.runtime_logging = params.get("enabled",True)
                        self.get_logger().info(f"Set runtime_logging to {self.runtime_logging}")
                    elif command == "step_speed_limitor":    
                        self.step_speed_limitor = params.get("value",None)
                        self.get_logger().info(f"Set step_speed_limitor to {self.step_speed_limitor}")
                    else:
                        self.get_logger().info(f"Command '{command}' not found")

            else:
                reading_messages = False

        # TODO: need to clean up redis when session terminates
        ctx.get_redis_client().set(f"SESSION_STATUS_{self._id}", json.dumps({
            'recording':self.continuous_recording_enabled,
            'runtime_logging': self.runtime_logging,
            "step_speed_limitor": self.step_speed_limitor,
            "last_update":self.last_command_update # for tracking if submission successful
        }))

        self.last_command_check = time.time()

    def set_render_fn(self, render_fn):
        self.render_fn = render_fn

    def get_run_info(self):
        data = {}
        data['run_meta'] = self.run_meta
        data['render_mode'] = self.render_mode
        data['env_run_meta'] = self.env_run_meta
        return data

    # ------------RESET--------------------
    def before_reset(self):
        if not self.initialized:
            self._initialize()
        if self.runtime_logging:
            self.recording = self.is_episode_recorded()
        return

    def after_reset(self, observation):
        self.prev_observation = observation
        self.prev_reward = None
        self.prev_done = False
        self.prev_info = None
        return

    # -------------STEP--------------------
    def before_step(self, action):
        # Log of previous env output + agent's action in response
        if self.runtime_logging and not self.prev_done:  # when done, gets logged after step
            self._log_step(
                observation=self.prev_observation,
                reward=self.prev_reward,
                done=self.prev_done,
                info=self.prev_info,
                action=action)
        self.step_counter += 1
        self.episode_step_counter += 1
        return

    def after_step(self, observation, reward, done, info, action):
        if self.step_speed_limitor:
            sleep_needed = 1/self.step_speed_limitor - (time.time() - self.last_tick)
            if sleep_needed > 0:
                time.sleep(sleep_needed)
            self.last_tick = time.time()

        if (time.time() - self.last_command_check) > self.command_check_interval:
            self.run_status_update()

        if self.config.max_steps_in_episode and self.episode_step_counter >= self.config.max_steps_in_episode:
            done = True

        if self.is_group_session:
            reward_sum = sum(reward.values())
            self.episode_reward_total += reward_sum
            self.reward_total += reward_sum
            done = done.get("__all__", False)
        else:
            self.episode_reward_total += reward or 0
            self.reward_total += reward or 0

        self.timer_agent_step_start = time.time()

        if done:
            if self.runtime_logging:
                self._log_step(
                    observation=observation,
                    reward=reward,
                    done=done,
                    info=info,
                    action=None)

            self.episode_counter += 1
            self.episode_step_counter = 0
            self.episode_reward_total = 0

        self.prev_action = action
        self.prev_observation = observation
        self.prev_reward = reward
        self.prev_done = done
        self.prev_info = info

        return

    def get_summary(self):

        cur_time = time.time()
        runtime = 0.0
        if self.session_start_time is not None:
            runtime = cur_time - self.session_start_time
        summary = {
            'runtime': runtime,
            'last_update': cur_time,
            "step_count": self.step_counter,
            "reward_total": self.reward_total,
            "episode_count": self.episode_counter,
            "episode_step_count": self.episode_step_counter,
            "episode_reward_total": self.episode_reward_total,
            "reward_mean_windowed": 0,
            "step_latency_mean_windowed": 0,
            "episode_reward_total_windowed": 0,
            "best_ep_reward_total": self.best_ep_reward_total,
            "best_ep_reward_total_ep_count": self.best_ep_reward_total_ep_count,
            "best_ep_reward_mean_windowed": self.best_ep_reward_mean_windowed,
            "best_ep_reward_mean_windowed_ep_count": self.best_ep_reward_mean_windowed_ep_count,
        }
        if self.prev_summary is None:
            self.prev_summary = summary
        step_delta = self.step_counter - self.prev_summary.get('step_count', 0)
        ep_delta = self.episode_counter - self.prev_summary.get('episode_count', 0)
        reward_delta = self.reward_total - self.prev_summary.get('reward_total', 0)
        latency_delta = cur_time - self.prev_summary.get('timestamp', cur_time)
        if step_delta > 0:
            summary['reward_mean_windowed'] = reward_delta / step_delta
            summary['step_latency_mean_windowed'] = latency_delta / step_delta
        else:
            summary['reward_mean_windowed'] = self.prev_summary['reward_mean_windowed']
            summary['step_latency_mean_windowed'] = self.prev_summary['step_latency_mean_windowed']
        if ep_delta > 0 and self.episode_counter > 3:
            summary['episode_reward_total_windowed'] = self.ep_log_buffer.mean('episode_reward_total', limit=3)
        else:
            summary['episode_reward_total_windowed'] = self.prev_summary['episode_reward_total_windowed']

        self.prev_summary = summary
        return self.prev_summary

    def _flush_step_log(self):
        if self.step_log_buffer.counter > 0:
            step_data = self.step_log_buffer.get_dict()
            ctx.get_store().save_session_logs(self._id, "step_stats", step_data)
            self.step_log_buffer.clear()
            self.save_summary()
        self.step_log_last_flush = time.time()

    def _flush_ep_log(self):
        if self.ep_log_buffer.counter > 0:
            ep_log_data = self.ep_log_buffer.get_dict()
            ctx.get_store().save_session_logs(self._id, "ep_stats", ep_log_data)
            self.ep_log_buffer.clear()
        self.ep_log_last_flush = time.time()

    def _log_step(self, observation, reward, done, info, action):
        """
        - add data to history
        - if done, add episode data to history
        - log step info for plotting
        - total_reward, mean_total_reward, 

        """
        cur_time = time.time()
        runtime = cur_time - self.session_start_time

        # collect step stats
        if (self.step_counter + 1) % self.config.step_log_freq == 0:

            latency_delta = cur_time - self.step_log_last_timestamp
            step_count_delta = self.step_counter - self.step_log_last_step_count

            self.step_log_mean_latency = 0
            try:
                self.step_log_mean_latency = latency_delta / step_count_delta
            except:
                pass

            self.step_log_last_step_count = self.step_counter
            self.step_log_last_timestamp = cur_time
            self.step_log_buffer.add((
                runtime,  # runtime
                self.step_counter,  # step_count,
                self.episode_counter,  # episode_count
                self.reward_total,  # reward_total
                self.step_log_mean_latency,
            ))

        # flush step logs
        if self.step_log_buffer.counter > 0 and ((cur_time - self.step_log_last_flush) > self.log_flush_interval_secs or self.step_log_buffer.is_full()):
            self._flush_step_log()

        # collect step stats
        if done:
            if (self.episode_counter % self.config.episode_log_freq == 0):
                self.ep_log_buffer.add((
                    runtime,  # timestamp
                    self.step_counter,
                    self.episode_counter,  # episode_count
                    self.episode_step_counter,  # episode_step_counter
                    self.episode_reward_total))  # episode_reward_total

            if self.best_ep_reward_total is None or self.episode_reward_total > self.best_ep_reward_total:
                self.best_ep_reward_total = self.episode_reward_total
                self.best_ep_reward_total_ep_count = self.episode_counter

            if self.episode_counter >= 2:
                self.last_ep_reward_mean_windowed = self.ep_log_buffer.mean('episode_reward_total', limit=3)
                if self.best_ep_reward_mean_windowed is None or self.last_ep_reward_mean_windowed > self.best_ep_reward_mean_windowed:
                    self.best_ep_reward_mean_windowed = self.last_ep_reward_mean_windowed
                    self.best_ep_reward_mean_windowed_ep_count = self.episode_counter

        # flush if time interval has passed or buffer is full
        if (done and self.ep_log_buffer.counter > 0 and (((cur_time - self.ep_log_last_flush) > self.log_flush_interval_secs))):  # or self.ep_log_buffer.is_full() TODO: FIX ME
            self._flush_ep_log()

        frame = None
        # Update Frame
        if self.frame_stream_enabled and ((cur_time - self.last_frame_update_timestamp) > self.frame_update_interval_secs):
            self.last_frame_update_timestamp = cur_time
            frame = self.render_frame()
            try:
                frame_size = frame.size
                en_shape = struct.pack('>II', frame_size[0], frame_size[1])
                encoded_image = en_shape + frame.tobytes()
                ctx.get_redis_client().publish("SESSION_STREAM_{}".format(self._id), encoded_image)
            except Exception as e:
                pass

        env_stats = None
        # Update Frame
        if self.frame_stream_enabled and ((cur_time - self.last_stats_update_timestamp) > self.stats_update_interval_secs):
            self.last_stats_update_timestamp = cur_time
            # Video Preview
            env_stats = self.render_stats()
            try:
                ctx.get_redis_client().publish("SESSION_STREAM_STATS_{}".format(self._id), json.dumps(env_stats))
            except Exception as e:
                pass

        ###################
        # RECORD ##########
        ###################
        if self.recording or self.continuous_recording_enabled:
            self.recording_last_timestamp = cur_time

            if frame is None and self.config.episode_record_preview_interval is not None and (self.episode_counter % self.config.episode_record_preview_interval == 0):
                frame = self.render_frame()

            # save frame
            if frame is not None:
                ctx.get_store().save_frame(
                    session_id=self._id,
                    episode_id=self.episode_counter,
                    step_id=self.episode_step_counter,
                    frame=frame)

            if env_stats is None and self.config.episode_record_preview_interval is not None and (self.episode_counter % self.config.episode_record_preview_interval == 0):
                env_stats = self.render_stats()

            self.history.add(
                (runtime,
                 self.step_counter,
                 self.episode_counter,
                 observation,
                 reward,
                 action,
                 done,
                 info,
                 self.reward_total,  # reward_total
                 self.episode_step_counter,  # episode_step_count
                 self.episode_reward_total,
                 env_stats))

            if done:
                episode_data = self.history.get_dict(limit=self.episode_step_counter + 1, limit_from_tail=True)

                ctx.get_store().save_episode_recording(
                    session_id=self._id,
                    episode_id=self.episode_counter,
                    values=episode_data)

    def close(self, state=None):

        self.step_log_buffer.add((
            time.time() - self.session_start_time,  # runtime
            self.step_counter,  # step_count,
            self.episode_counter,  # episode_count
            self.reward_total,  # reward_total
            self.step_log_mean_latency
        ))

        self._flush_step_log()
        self._flush_ep_log()
        self.save_summary()

        if state is None:
            if self.is_complete():
                state = STATE_COMPLETED
            else:
                state = STATE_TERMINATED

        self.change_status(state)
        self.get_logger().info("Ending Session")
        self.running = False
        return state

    def is_episode_recorded(self):

        recorded_ep = (
            (self.config.episode_record_freq is not None) and (self.config.episode_record_freq != 0) and
            ((self.episode_counter + 1) % self.config.episode_record_freq == 0))

        return recorded_ep and (self.recording_interval_secs and ((time.time() - self.recording_last_timestamp) > self.recording_interval_secs))

    def render_stats(self):
        try:
            if self.render_stats_enabled and self.render_fn:
                return self.render_fn(mode="stats")
            else:
                return {}
        except Exception as e:
            msg = "Stats Generation Failed. Disabling Stats for session. Exception: {},\n{}".format(e, traceback.format_exc())
            self.get_logger().error(msg)
            self.render_stats_enabled = False
            return {'error': msg}
        except NotImplementedError as e:
            msg = "Render not implemented. Disabling Stats for session. Exception: {}".format(e)
            self.get_logger().error(msg)
            self.render_stats_enabled = False
            return {'error': msg}

    def render_frame(self):
        try:
            if self.render_frame_enabled and self.render_fn:
                return render_env_frame_as_image(self.render_fn, self.render_mode)
            else:
                return None
        except Exception as e:
            msg = "Preview Generation Failed. Disabling Preview for session. Exception: {},\n{}".format(e, traceback.format_exc())
            self.get_logger().error(msg)
            self.render_frame_enabled = False
            return msg
        except NotImplementedError as e:
            msg = "Render not implemented. Disabling Preview for session. Exception: {}".format(e)
            self.get_logger().error(msg)
            self.render_frame_enabled = False
            return msg

    def is_complete(self):
        return not self.running or ((self.config.max_episodes) and (self.episode_counter > self.config.max_episodes)) or \
            ((self.config.max_steps) and (self.step_counter > self.config.max_steps))
