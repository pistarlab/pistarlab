import json
import logging
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from gym import spaces

from pistarlab.agent import Agent
from .predictors_mlp import (AC2ModelMLP, PolicyEstimatorMLP,
                                         PolicyModel, ValueEstimatorMLP,
                                         ValueModel)
from .transformers import (IntToOneHotTransformer,
                                       MirrorTransformer,
                                       NpArrayFlattenTransformer, Transformer,
                                       build_action_transformer,
                                       build_observation_transformer,
                                       prep_frames)
from pistarlab.meta import STATE_COMPLETED, STATE_RUNNING
from pistarlab.session_config import RLSessionConfig
from pistarlab.session_env import RLMultiSessionEnv
from pistarlab.task_runner import AgentTaskRunner


class A2C(Agent):
    pass


class A2CTaskRunner(AgentTaskRunner):

    def run(self):
        task = self.get_task()
        agent = task.get_agent()

        task_id = task.get_id()
        task_config = task.get_config()

        env_spec_id = task_config['env_spec_id']
        env_kwargs = task_config['env_kwargs']

        session_config = task_config['session_config']
        agent_run_config = agent.get_config(task_config['agent_run_config'])

        batch_size = task_config.get('batch_size', 1)
        use_remote_client = task_config.get('use_remote_client', False)

        start_time = time.time()

        env = RLMultiSessionEnv(
            env_spec_id=env_spec_id,
            env_kwargs=env_kwargs,
            config=RLSessionConfig(**session_config),
            agent_id=agent.get_id(),
            agent_run_config=agent_run_config,
            batch_size=batch_size,
            task_id=task_id,
            use_remote_client=use_remote_client,
            timeout_abort_check_callback=lambda: task.get_status() != STATE_RUNNING)

        if len(env.players) != 1:
            raise Exception("UNDER DEVELOPMENT: Only supports single player envs")

        observation_space = list(env.observation_spaces.values())[0]
        action_space = list(env.action_spaces.values())[0]

        discount_factor = 0.99
        p_learning_rate = 0.0001
        v_learning_rate = 0.0001

        ob_transformer = build_observation_transformer(observation_space, for_cnn=False)
        action_transformer = build_action_transformer(action_space)

        model_p = PolicyModel(
            input_size=ob_transformer.get_shape(),
            action_num=action_transformer.get_shape(),
            hidden_nodes=256,
            learning_rate=p_learning_rate)

        model_v = ValueModel(
            input_size=ob_transformer.get_shape(),
            action_num=action_transformer.get_shape(),
            hidden_nodes=256,
            learning_rate=v_learning_rate)

        policy_estimator = PolicyEstimatorMLP(model_p)
        value_estimator = ValueEstimatorMLP(model_v)

        obv_prev = None
        action = None
        save_freq_in_seconds = 60
        last_save_timestamp = time.time()

        player_id = env.players[0]
        step_count = 0
        ep_count = 0
        running = True
        done = True

        batch_size = 32
        obv_prev_batch = []
        value_prev_updated_batch = []
        advantage_batch = []
        action_batch = []

        update_count = 0


        try:

            while running:
                if done:
                    obs = env.reset()
                    rews, dones, infos = {}, {}, {}
                    for k in obs.keys():
                        rews[k] = None
                        dones[k] = False
                        infos[k] = {}
                    if step_count > 0:
                        ep_count += 1
                    dones["__all__"] = False

                    obv_prev = None
                else:
                    obs, rews, dones, infos = env.step({player_id: action})
                    step_count += 1

                # TODO: Replace with agent code
                #action_dict = {player_id: env.action_spaces[player_id].sample() for player_id in obs.keys()}
                # action = action_dict.get(player_id)

                ob = obs[player_id]
                reward = rews[player_id]
                # done = dones[player_id]
                done = dones.get("__all__", False)
                info = infos[player_id]
                p_loss = 0
                v_loss = 0
                obv = ob_transformer.transform(ob)
                if update_count % 10 == 0:
                    self.get_logger().info(f"ob: {ob.shape}, obv:{obv.shape}, reward:{reward}, done:{done}")

                value_pred = value_estimator.predict(obv)

                if obv_prev is not None:  # Don't run on first step of episode

                    # Value of previous observation
                    value_prev_prediction = value_estimator.predict(obv_prev)

                    if done:
                        value_prev_updated = reward

                        advantage = reward - value_prev_prediction
                    else:
                        value_prev_updated = reward + (discount_factor * value_pred)

                        advantage = value_prev_updated - value_prev_prediction

                    # Update networks
                    if len(obv_prev_batch) >= batch_size:
                        v_loss += value_estimator.update(
                            np.array(obv_prev_batch),
                            np.array(value_prev_updated_batch))

                        p_loss += policy_estimator.update(
                            np.array(obv_prev_batch),
                            np.array(advantage_batch),
                            action_transformer.transform(np.array(action_batch)))

                        if update_count % 100 == 0:
                            p_loss = p_loss/100
                            v_loss = v_loss/100
                            self.get_logger().info(f"Step Count: {step_count}, p_loss:{p_loss}, v_loss: {v_loss}")
                            agent.log_stat_dict(
                                task_id=task.get_id(),
                                data={'v_loss': v_loss, 'p_loss': p_loss})
                            p_loss = 0
                            v_loss = 0

                        update_count += 1

                        obv_prev_batch = []
                        value_prev_updated_batch = []
                        advantage_batch = []
                        action_batch = []
                    else:
                        obv_prev_batch.append(obv_prev)
                        value_prev_updated_batch.append(value_prev_updated)
                        advantage_batch.append(advantage)
                        action_batch.append(action)

                action, _ = policy_estimator.predict_choice(obv)
                obv_prev = obv

                if env.is_complete() or not self.is_running():
                    running = False

                # make checkpoint if needed
                if not running or ((time.time() - last_save_timestamp) > save_freq_in_seconds):
                    self.get_logger().info(f"Saving Agent Stats: Step Count: {step_count}")
                    # checkpoint_data = {}
                    # checkpoint_data[F_TIMESTAMP] = time.time()
                    # agent.save_state(state=trainer.get_weights()[DEFAULT_POLICY_ID], meta=checkpoint_data)
                    agent.flush_stats()
                    last_save_timestamp = time.time()

        finally:
            agent.flush_stats()
            pre_close_delta = time.time() - start_time
            delta = time.time() - start_time
            self.get_logger().info("Runtime in seconds: {}, Step_Count:{}, steps_per_second: {} preclose: {}".format(delta, step_count, step_count / delta, step_count / pre_close_delta))

        env.close()
        exit_state = STATE_COMPLETED

        return {}, exit_state
