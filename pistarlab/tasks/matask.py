
from .. import ctx, Agent, Task
import time
import logging
from ..task import Task, AgentTask
from ..task_runner import TaskRunner
from ..utils.pyson import pyson

from ..meta import *
from ..util_funcs import *
from ..remote_env import RemoteMultiAgentEnvServer
from ..utils.serialize import space_to_pyson

from ..session import RLSession
from ..session_config import RLSessionConfig

def get_agent_space_map(player_to_agent_map,player_space_map):
    agent_ids = set(player_to_agent_map.values())
    agent_space_map= {k:{} for k in agent_ids}
    for player_id, space in player_space_map.items():
        agent_id = player_to_agent_map[player_id]
        agent_space_map[agent_id][player_id] = space
    return agent_space_map  

class MultiAgentRunner(TaskRunner):

    spec_id = 'multiagent'
    displayed_name = ""
    extension_id ="builtin"
    version = "0.0.1.dev0"
    config = {
        'agents': {},
        'env_spec_id': None,
        'env_kwargs': {},
        'session_config': {},
        'player_assignments': {}
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.task:Task = self.get_task()
        config = self.task.get_config()
        var_lookup = {}
        self.config = pyson(config, var_lookup=var_lookup)
        session_config = self.config['session_config']

        env_spec_id = self.config['env_spec_id']
        env_kwargs = self.config['env_kwargs']
        player_to_agent_map = self.config['player_assignments']
        task_id = self.task.get_id()

        self.session = RLSession(
                config=RLSessionConfig(**session_config),
                task_id=self.task.get_id(),
                agent_id=None,
                agent_run_config=None,
                env_spec_id=env_spec_id,
                session_type=RL_MULTIPLAYER_MULTIAGENT_SESS,
                run_meta={'player_to_agent_map':player_to_agent_map},
                persists_at_creation=True)

        self.task.set_primary_session_by_id(self.session.get_id())

        self.agent_tasks = {}

        def abort_task_check_callback():
            if self.task.get_status() != "RUNNING":
                return True
            else:
                for task in self.agent_tasks.values():        
                    if task.get_status() in [STATE_TERMINATED, STATE_ABORTED]:
                        self.get_logger().info("Child task not running. Terminating session multiagent session")
                        return True
                return False
        
        self.env_server = RemoteMultiAgentEnvServer(
            session = self.session,
            task_id=task_id,
            env_spec_id= env_spec_id, 
            env_kwargs = env_kwargs,
            player_to_agent_map = player_to_agent_map,
            timeout_abort_check_callback= abort_task_check_callback)

        self.session.render_fn =lambda mode:self.env_server.env.render(mode = mode)

        obs_spaces = {k: space_to_pyson(v) for k,v in self.env_server.observation_spaces.items()}
        action_spaces = {k: space_to_pyson(v) for k,v in self.env_server.action_spaces.items()}

        agent_observation_space_map = get_agent_space_map(player_to_agent_map,obs_spaces)
        agent_action_space_map = get_agent_space_map(player_to_agent_map,action_spaces)

        
        for agent_data in self.config['agents']:
            agent_id = agent_data['ident']
            agent = Agent.load(id=agent_id)

            session_config = agent_data['session_config']
            agent_env_kwargs={}
            agent_env_kwargs['task_id'] = task_id
            agent_env_kwargs['session_id'] = self.session.get_id()
            agent_env_kwargs['observation_spaces'] = agent_observation_space_map[agent_id]
            agent_env_kwargs['action_spaces'] = agent_action_space_map[agent_id]

            task = AgentTask.create(
                env_spec_id=env_spec_id,
                env_kwargs = agent_env_kwargs,
                agent = agent,
                use_remote_client = True,
                session_config=session_config,
                batch_size=1,
                parent_task_id= self.task.get_id())
            self.agent_tasks[task.get_id()] = task

        self.max_episodes = session_config['max_episodes']
        self.max_steps = session_config['max_steps']
        self.status_check_freq_secs = 4

    def run(self):

        task_id = self.task.get_id()
        
        for task in self.agent_tasks.values():
            task.run()

        last_check = 0
        exit_state = None
        terminated = False
        done = False
        self.get_logger().info("Running task {}".format(self.task.get_id()))
        try:

            while not done:
                self.env_server.run_step()
                # Check for terminination request
                if not done and (time.time() - last_check > self.status_check_freq_secs):
                    if self.task.get_status() != STATE_RUNNING:
                        done = True
                        terminated = True
                        self.get_logger().info("Terminating work for {} as requested".format(task_id))
                    else:
                        
                        for task in self.agent_tasks.values():
                            
                            if task.get_status() in [STATE_TERMINATED, STATE_ABORTED]:
                                done = True 
                                terminated = True
                                self.get_logger().info("Child task not running. Terminating session multiagent session")
                                break


                    last_check = time.time()

                    # Check timestep and episode limits
                    if not done and self.max_steps and self.env_server.step_counter >= self.max_steps:
                        done = True
                        self.get_logger().info(f"Max Steps Reached for {task_id} , max_steps/reached: {self.max_steps}/{self.env_server.step_counter}")
                    elif not done and self.max_episodes and self.env_server.episode_counter >= self.max_episodes:
                        done = True
                        self.get_logger().info("Max Episodes Reached for {}".format(task_id))

            if terminated:
                exit_state= STATE_TERMINATED
            else:
                exit_state = STATE_COMPLETED

        except Exception as e:
            logging.error(e)
        finally:
            try:
                self.env_server.close()
            except Exception as e:
                pass
            for id, task in self.agent_tasks.items():
                try:
                    task.shutdown()
                except Exception as e:
                    pass

        if terminated:
            exit_state = STATE_TERMINATED
        else:
            exit_state = STATE_COMPLETED
            
        return {}, exit_state
