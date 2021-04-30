import logging
import time

import gym

from . import ctx
from .dbmodels import EnvSpecModel
from .remote_conn import ClientConn, ServerConn
from .utils.env_helpers import get_wrapped_env_instance

from .session import RLSession


class RemoteMultiAgentEnvClient:

    def __init__(self, task_id, session_id, agent_id, observation_spaces, action_spaces, timeout_abort_check_callback):
        self.agent_id = agent_id
        self.players = list(observation_spaces.keys())
        self.num_players = len(observation_spaces)
        self.observation_spaces = observation_spaces
        self.action_spaces = action_spaces
        self.pending_response = None

        self.conn = ClientConn(
            task_id=task_id,
            client_id=self.agent_id,
            timeout_abort_check_callback=timeout_abort_check_callback)

    def reset(self):
        if self.pending_response is None:
            response = self.conn.get()
        else:
            response = self.pending_response
            self.pending_response = None
        msginfo = response['msginfo']
        obs, rews, dones, infos = response['data']
        return obs

    def step(self, action_dict):
        self.step_action(action_dict)
        return self.step_get_observation()

    def step_action(self, action_dict):
        packet = {
            'sender': self.agent_id,
            'msginfo': {'type': "action"},
            'data': action_dict}

        # Send Observations to agents
        self.conn.put(packet)

    def step_get_observation(self):
        # get response
        if self.pending_response is None:
            response = self.conn.get()
        else:
            response = self.pending_response
            self.pending_response = None
        obs, rews, dones, infos = response['data']
        msginfo = response['msginfo']

        all_done = all(dones.values())
        dones['__all__'] = all_done
        return obs, rews, dones, infos

    def stats(self, *args, **kwargs):
        return None

    def render(self, *args, **kwargs):
        # logging.info(f"MM:{self.agent_id} Sending Rendering request")
        packet = {
            'sender': self.agent_id,
            'msginfo': {'type': 'render'},
            'data': {'args': args, 'kwargs': kwargs}
        }

        self.conn.put(packet)
        while True:
            response = self.conn.get()
            # logging.info(f"MM:{self.agent_id} Render response:{response['msginfo']}")
            if response['msginfo']['type'] == "render":
                return response['data']
            else:
                if self.pending_response is None:
                    self.pending_response = response
                else:
                    msg = "ERROR multiple responses pending processing"
                    logging.error(msg)
                    raise Exception("ERROR multiple responses pending processing")

    def close(self):
        logging.info(f"MM:{self.agent_id} Closing Connection")
        packet = {
            'sender': self.agent_id,
            'msginfo': {'type': 'close'},
            'data': {}
        }

        self.conn.close()


class RemoteSingleAgentEnvClient(RemoteMultiAgentEnvClient, gym.Env):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        gym.Env().__init__()
        assert(len(self.players) == 1), "Number of agents must be 1, found {}".format(len(self.players))
        self.player_id = self.players[0]
        self.observation_space = self.observation_spaces[self.player_id]
        self.action_space = self.action_spaces[self.player_id]

    def reset(self):
        return super().reset()[self.player_id]

    def step(self, action):
        obs, rews, dones, infos = super().step({self.player_id: action})
        return obs[self.player_id], rews[self.player_id], dones[self.player_id], infos[self.player_id]


class RemoteMultiAgentEnvServer:

    def __init__(
            self,
            session: RLSession,
            player_to_agent_map={},
            timeout_abort_check_callback=lambda: False,
            env=None, *args, **kwargs):

        self.session = session

        self.env_spec_id = self.session.env_spec_id

        spec_model = self.get_spec_env_dbmodel()

        env_entry_point = spec_model.entry_point

        if env is None:
            wrappers = spec_model.config['default_wrappers']
            env = get_wrapped_env_instance(
                entry_point=env_entry_point,
                kwargs=self.session.env_kwargs,
                wrappers=wrappers)

        self.env = env

        self.observation_spaces = self.env.observation_spaces
        self.action_spaces = self.env.action_spaces

        self.player_to_agent_map = player_to_agent_map

        self.action_dict = {}
        self.reset_flag = True
        self.task_id = self.session.task_id

        self.step_counter = 0
        self.episode_counter = 0
        self.steps_in_episode_counter = 0
        self.conn = ServerConn(
            task_id=self.task_id,
            client_ids=list(self.player_to_agent_map.values()),
            timeout_abort_check_callback=timeout_abort_check_callback)

        self.obs = {}
        self.rews = {}
        self.dones = {}
        self.infos = {}
        self.actions = {}

        self.flush_interval_secs = 1.0
        self.last_summary_flush = 0

    def get_spec_env_dbmodel(self) -> EnvSpecModel:
        return ctx.get_dbsession().query(EnvSpecModel).get(self.env_spec_id)

    def process_message_from_client(self, result, actions, incoming_agent_msgs):
        # logging.info(f"Server Received: {result}")
        msg_type = result['msginfo']['type']
        if msg_type == "action":
            incoming_agent_msgs.remove(result['sender'])
            for player_id, action in result['data'].items():
                actions[player_id] = action
        elif msg_type == "render":

            agent_id = result['sender']
            render_kwargs = result['data']['kwargs']
            try:
                render_output = self.env.render(
                    mode=render_kwargs.get("mode"),
                    player_id=render_kwargs.get("player_id"))
                packet = {'msginfo': {'type': "render"}, 'data': render_output}
            except Exception as e:
                logging.error(e)
                packet = {'msginfo': {'type': "render"}, 'data': None}
            # logging.info(f"Server Sending: render package:{packet}")
            self.conn.put(agent_id, packet)
        elif msg_type == "close":
            agent_id = result['sender']
            msg = f"Connection closed from client: {agent_id}. Aborting Sessions"
            logging.info(msg)
            self.close()
            raise Exception(msg)
        else:
            msg = f"Unknown type {msg_type} from client: {result['sender']}. Aborting Sessions"
            logging.error(msg)
            self.close()
            raise Exception(msg)

    def get_actions(self):
        """
        sends observations and waits for actions
        """

        # Bundle Observations by agent_id
        #'__all__':self.dones.get('__all__',False)
        agent_out_map = {}
        for key in self.obs.keys():
            # if self.dones[key]:
            #     continue
            agent_id = self.player_to_agent_map.get(key)
            packet = agent_out_map.get(agent_id, {'msginfo': {'type': "obs"}, 'data': ({}, {}, {}, {})})
            kobs, krews, kdones, kinfos = packet['data']
            kobs[key] = self.obs[key]
            krews[key] = self.rews[key]
            kdones[key] = self.dones[key]
            kinfos[key] = self.infos[key]
            packet['data'] = (kobs, krews, kdones, kinfos)
            agent_out_map[agent_id] = packet

        # Send Observations to agents
        incoming_agent_msgs = set(agent_out_map.keys())
        for agent_id, packet in agent_out_map.items():
            self.conn.put(agent_id, packet)
            if all(packet['data'][2].values()):
                incoming_agent_msgs.remove(agent_id)

        # Get Actions from those same agents
        actions = {}
        # Process queued messages
        for result in self.conn.get_all():
            self.process_message_from_client(result, actions, incoming_agent_msgs)

        # Process remaining actions
        while len(incoming_agent_msgs) > 0:
            result = self.conn.get()
            self.process_message_from_client(result, actions, incoming_agent_msgs)
        return actions

    def reset(self):
        ob_dict = self.env.reset()
        self.steps_in_episode_counter = 0
        if self.step_counter > 0:
            self.episode_counter += 1
        return ob_dict

    def get_summary(self):
        return {
            'step_count': self.step_counter,
            'episode_counter': self.episode_counter,
            'steps_in_episode_counter': self.steps_in_episode_counter
        }

    def run_step(self):
        if self.reset_flag:
            self.session.before_reset()
            self.obs = self.reset()
            self.session.after_reset(self.obs)
            self.rews, self.dones, self.infos = {}, {}, {}
            for key in self.obs.keys():
                self.rews[key] = None
                self.dones[key] = False
                self.infos[key] = {'reset': True}

            self.reset_flag = False
        else:
            self.session.before_step(self.action_dict)
            self.obs, self.rews, self.dones, self.infos = self.env.step(self.action_dict)
            self.session.after_step(self.obs, self.rews, self.dones, self.infos, self.action_dict)

        self.step_counter += 1
        self.steps_in_episode_counter += 1

        self.action_dict = self.get_actions()
        if self.dones.get('__all__', False):
            self.reset_flag = True

        # save session stats
        cur_time = time.time()
        if self.reset_flag or (((cur_time - self.last_summary_flush) > self.flush_interval_secs)):
            self.session.save_summary()
            self.last_summary_flush = cur_time

    def close(self):
        self.session.close()
        self.conn.close()
        self.env.close()
