from typing import Any
from ..utils.misc import gen_uid
import ray
from gym import Wrapper
import gym
class Session(Wrapper):

    def __init__(self, session_def, agent_info, shared_state):
        env = gym.make("CartPole-v1")
        self.uid = gen_uid()
        self.max_steps = 1000
        self.tasker_id = None
        self.shared_state = shared_state
        Wrapper.__init__(env=env)

    def get_uid(self):
        return self.uid

    def register_tasker(self,tasker_id):
        self.tasker_id


    def reset(self):
        return self.env.reset(self)

    def step(self,action):
        return self.env.step(action)

    def is_complete(self):
        return self.step_count >= self.max_steps

from dataclasses import dataclass



# TODO: VectorEnv/ MultiTasker / Shared Env
class Executor:


    def __init__(self):
        self.tasker = None
        self.session = None
        self.shared_state = None

    def submit_task(self, agent_info, tasker_id, tasker_def, session_def):

        self.shared_state = {'counter':0} # TODO: get from session_def ... pydef??
        
        # TODO, maybe vector envs?
        self.session = Session(
            session_def= session_def, 
            agent_info= agent_info, 
            shared_state=self.shared_state)

        # Update with env variable
        tasker_def = agent_info['tasker_def']
        tasker_def['observation_space'] = self.session.observation_space
        tasker_def['action_space'] = self.session.action_space
        
        agent_config = AgentTaskerConfig.instance_from_def(tasker_def)
        self.tasker = AgentTasker.instance_from_config(
                tasker_id=tasker_id, 
                agent_info = agent_info, 
                config = agent_config)

        session_ids = self.session.get_uid()
                
        return session_ids

    def run(self):
        # TODO, run in thread, switch in thread for multiple taskers??
        # Use locks to ensure alternating threads or dont
        self.tasker.run(self.session)
        return True

class AgentConfig:

    def __init__(self):
        pass

# Actor Service/ All Actions are Non blocking
class Agent:

    def __init__(self, uid, dict_def):
        self.uid = uid
        self.dict_def= dict_def
        self.tasker_count = 0
        self.tasker_lookup = {}
        
    def get_uid(self):
        return self.uid   
    
    def get_instance_info(self):
        return {'uid':self.get_uid(),'dict_def':self.dict_def}

    def submit_tasker_update(self,tasker_id, data):
        print("Recieved Tasker Update: {}:  {} ".format(tasker_id,data))

    def add_to_submission_log(self,tasker_id,executor,tasker_def, session_def, session_ids):
        # Save to database
        self.tasker_lookup[tasker_id] = {
            'executor': executor, 
            'tasker_def': tasker_def, 
            'session_def': session_def,
            'session_ids':session_ids}
        self.tasker_count +=1


    def submit_task(self,session_def):
        # 
        # This function parses session_def and handles details around how and what is submitted

        executor = Executor.remote()
        tasker_id = self.tasker_count
        tasker_def = self.dict_def['tasker_def']
        session_ids_future = executor.submit_task.remote(
            tasker_id = tasker_id,
            tasker_def = tasker_def,
            agent_info = self.get_instance_info(), 
            session_def = session_def)
        session_ids = ray.get(session_ids_future)
        self.add_to_submission_log(
            tasker_id=tasker_id,
            executor=executor,
            tasker_def=tasker_def, 
            session_def=session_def, 
            session_ids=session_ids)
        print("Submission Info: {}".format())

# Serializable Config
class AgentTaskerConfig:

    @classmethod
    def instance_from_def(cls,tasker_def):

        return cls(
            observation_space = tasker_def['observation_space'],
            action_space = tasker_def['action_space'])

    def __init__(self, observation_space, action_space):
        self.action_space = action_space
        self.observation_space = observation_space

# Non Actor
class AgentTasker:

    @classmethod
    def instance_from_config(
        cls,
        config: AgentTaskerConfig, 
        agent_info = None,
        tasker_id = None):

        agent:Agent = ray.get_actor(name = agent_info['uid']) # Get Agent Reference from agent_info

        return cls(config=config, tasker_id=tasker_id, agent=agent)

    def __init__(self, config: AgentTaskerConfig, tasker_id:str, agent:Agent):
        self.tasker_id = tasker_id
        self.agent = agent
        self.config = config
        self.agent.submit_tasker_update.remote(tasker_id=self.tasker_id,data={'msg':'Hi'})

    def run(self,session):
        self.agent.submit_tasker_update.remote(tasker_id=self.tasker_id,data={'msg':'starting session {}'.format(session.get_uid())})

        ob = session.reset()
        while not session.is_complete():
            session.step(self.config.action_space.sample())
        self.agent.submit_tasker_update.remote(tasker_id=self.tasker_id,data={'bye'})





