from gym.spaces import Box, Discrete
import numpy as np

# TODO: need standard interface defined



class MultiAgentTestParallelEnv:
    
    def __init__(self,num_players=2):
        self.num_players = num_players
        self.max_num_players = num_players
        self.players = [f"player_{i}" for i in range(self.num_players)] 
        self.possible_players = self.players
        self.observation_spaces = { f"player_{i}":Box(high=20,low=0,shape=(20,20),dtype=np.uint8) for i in range(self.num_players)}
        self.action_spaces = { f"player_{i}":Box(high=20,low=0,shape=(10,),dtype=np.uint8) for i in range(self.num_players)}
        

        self.num_steps_before_reset = 10
        self.step_counter = 0
        self.reset_needed = True
        self.steps_in_ep = 0
        self.ep_counter = 0
        self.done_players = set()
        
    def get_player_id(self,i):
        return f"player_{i}"
    
    def reset(self):
        self.reset_needed = False
        self.steps_in_ep = 0
        self.num_steps_before_reset +=10
        self.ep_counter +=1
        self.done_players = set()
        obs = {}
        for i in range(self.num_players):
            obs[self.get_player_id(i)] =  self.observation_spaces[self.get_player_id(i)].sample()
        return obs
    
    def step(self,action):
        assert(not self.reset_needed), "Reset needs to be called"
            
        obs,rews,dones,infos = {}, {}, {}, {}
        self.step_counter +=1
        self.steps_in_ep +=1
        if self.steps_in_ep >= self.num_steps_before_reset:
            dones['__all__'] = True
            self.reset_needed = True
        else:
            dones['__all__'] = False

        for i in range(self.num_players):
            if not i in self.done_players:
                now_done = dones['__all__'] or (i ==0 and (self.steps_in_ep >= self.num_steps_before_reset/2))
                if now_done:
                    self.done_players.add(i)
                obs[self.get_player_id(i)] =  self.observation_spaces[self.get_player_id(i)].sample()
                rews[self.get_player_id(i)] = 1
                dones[self.get_player_id(i)] = now_done
                infos[self.get_player_id(i)] = {}

        return obs,rews,dones,infos


    def close(self):
        pass

    def render(self,*args,**kwargs):
        return None


class MultiAgentTestAlternatingSlotEnv:
    
    def __init__(self,num_players=2):
        self.num_players = num_players
        self.max_num_players = num_players
        self.players = [f"player_{i}" for i in range(self.num_players)] 
        self.possible_players = self.players
        self.observation_spaces = { f"player_{i}":Box(high=20,low=0,shape=(10,),dtype=np.uint8) for i in range(self.num_players)}
        self.action_spaces = { f"player_{i}":Box(high=20,low=0,shape=(10,),dtype=np.uint8) for i in range(self.num_players)}
        
        self.whos_turn = 0
        self.turns = 0
        
    def get_player_id(self):
        return f"player_{self.whos_turn}"
    
    def reset(self):
        self.whos_turn = 0 #random.randint(0,self.num_players)
        self.turns =0
        return {self.get_player_id(): self.observation_spaces[self.get_player_id()].sample()}
    
    def step(self,action):
        self.whos_turn = (self.whos_turn + 1) % self.num_players
        self.turns +=1
        ob, rew,done,info = self.observation_spaces[self.get_player_id()].sample(),1,False,{}
        obs = {self.get_player_id(): ob}
        rews = {self.get_player_id(): rew}
        dones = {self.get_player_id(): done}
        infos = {self.get_player_id(): info}
        return obs,rews,dones,infos
 
    def close(self):
        pass