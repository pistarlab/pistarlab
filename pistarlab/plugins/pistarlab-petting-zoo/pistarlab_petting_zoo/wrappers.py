# Copied From: https://github.com/ray-project/ray/blob/master/rllib/env/pettingzoo_env.py#L102
# from pistarlab.common import MAEnv

class PettingZooAECWrapper:

    def __init__(self, env):
        """
        Parameters:
        -----------
        env:  AECenv object.
        """
        self.env = env

        # agent idx list
        self.env.reset()
        self.players = self.env.agents

        # Get dictionaries of obs_spaces and act_spaces
        self.observation_spaces = self.env.observation_spaces
        self.action_spaces = self.env.action_spaces
        self.num_players = self.env.num_agents
        self.possible_players = self.env.possible_agents
        self.max_num_players = self.env.max_num_agents

        # Get first observation space, assuming all agents have equal space
        self.observation_space = self.observation_spaces[self.players[0]]

        # Get first action space, assuming all agents have equal space
        self.action_space = self.action_spaces[self.players[0]]

        assert all(obs_space == self.observation_space
                   for obs_space
                   in self.env.observation_spaces.values()), \
            "Observation spaces for all agents must be identical. Perhaps " \
            "SuperSuit's pad_observations wrapper can help (useage: " \
            "`supersuit.aec_wrappers.pad_observations(env)`"

        assert all(act_space == self.action_space
                   for act_space in self.env.action_spaces.values()), \
            "Action spaces for all agents must be identical. Perhaps " \
            "SuperSuit's pad_action_space wrapper can help (useage: " \
            "`supersuit.aec_wrappers.pad_action_space(env)`"

        self.rewards = {}
        self.dones = {}
        self.obs = {}
        self.infos = {}

        _ = self.reset()

    def _init_dicts(self):
        # initialize with zero
        self.rewards = dict(zip(self.players, [0 for _ in self.players]))
        # initialize with False
        self.dones = dict(zip(self.players, [False for _ in self.players]))
        self.dones["__all__"] = False

        # initialize with None info object
        self.infos = dict(zip(self.players, [{} for _ in self.players]))

        # initialize empty observations
        self.obs = dict(zip(self.players, [None for _ in self.players]))

    def reset(self):
        """
        Resets the env and returns observations from ready agents.
        Returns:
            obs (dict): New observations for each ready agent.
        """
        # 1. Reset environment; agent pointer points to first agent.
        self.env.reset()

        # 2. Copy agents from environment
        self.players = self.env.agents

        # 3. Reset dictionaries
        self._init_dicts()

        # 4. Get initial observations
        for player_id in self.players:

            # For each agent get initial observations
            self.obs[player_id] = self.env.observe(player_id)

        return self.obs

    def step(self, action_dict):
        """
        Executes input actions from RL agents and returns observations from
        environment agents.
        The returns are dicts mapping from agent_id strings to values. The
        number of agents in the env can vary over time.
        Returns
        -------
            obs (dict): New observations for each ready agent.
            rewards (dict): Reward values for each ready agent. If the
                episode is just started, the value will be None.
            dones (dict): Done values for each ready agent. The special key
                "__all__" (required) is used to indicate env termination.
            infos (dict): Optional info values for each agent id.
        """
        stepped_players = set()
        while (self.env.agent_selection not in stepped_players
               and self.env.dones[self.env.agent_selection]):
            players = self.env.agent_selection
            self.env.step(None)
            stepped_players.add(players)
        stepped_players = set()
        while (self.env.agent_selection not in stepped_players):
            players = self.env.agent_selection
            assert players in action_dict or self.env.dones[players], \
                "Live environment agent is not in actions dictionary"
            self.env.step(action_dict[players])
            stepped_players.add(players)

        assert all(agent in stepped_players or self.env.dones[agent]
                   for agent in action_dict), \
            "environment has a nontrivial ordering, and cannot be used with"\
            " the POMGameEnv wrapper"

        self.obs = {}
        self.rewards = {}
        self.dones = {}
        self.infos = {}

        # update self.agents
        self.players = list(action_dict.keys())

        for players in self.players:
            self.obs[players] = self.env.observe(players)
            self.dones[players] = self.env.dones[players]
            self.rewards[players] = self.env.rewards[players]
            self.infos[players] = self.env.infos[players]

        self.dones["__all__"] = all(self.env.dones.values())

        return self.obs, self.rewards, self.dones, self.infos

    def render(self, mode="human",player_id=None):
        return self.env.render(mode=mode)

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        self.env.seed(seed)
