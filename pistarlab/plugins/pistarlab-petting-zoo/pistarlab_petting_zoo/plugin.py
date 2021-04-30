from pistarlab.meta import RL_MULTIPLAYER_ENV
from pistarlab.wrappers import default
import gym
import logging
from pistarlab import ctx


from pettingzoo.atari import boxing_v1 as boxing
from pettingzoo.atari import combat_plane_v1 as combat_plane
from pettingzoo.atari import combat_tank_v1 as combat_tank
from pettingzoo.atari import double_dunk_v2 as double_dunk
from pettingzoo.atari import entombed_competitive_v2 as entombed_competitive
from pettingzoo.atari import entombed_cooperative_v2 as entombed_cooperative
from pettingzoo.atari import flag_capture_v1 as flag_capture
from pettingzoo.atari import ice_hockey_v1 as ice_hockey
from pettingzoo.atari import joust_v2 as joust
from pettingzoo.atari import mario_bros_v2 as mario_bros
from pettingzoo.atari import maze_craze_v2 as maze_craze
from pettingzoo.atari import othello_v2 as othello
from pettingzoo.atari import pong_v1 as pong

from pettingzoo.atari import space_invaders_v1 as space_invaders
from pettingzoo.atari import space_war_v1 as space_war
from pettingzoo.atari import surround_v1 as surround
from pettingzoo.atari import tennis_v2 as tennis
from pettingzoo.atari import video_checkers_v2 as video_checkers
from pettingzoo.atari import wizard_of_wor_v2 as wizard_of_wor
from pettingzoo.atari import warlords_v2 as warlords

from pettingzoo.classic import chess_v2 as chess
from pettingzoo.classic import rps_v1 as rps
from pettingzoo.classic import rpsls_v1  as rpsls
from pettingzoo.classic import connect_four_v1 as connect_four
from pettingzoo.classic import tictactoe_v1 as tictactoe
from pettingzoo.classic import leduc_holdem_v1 as leduc_holdem
from pettingzoo.classic import mahjong_v1 as mahjong
from pettingzoo.classic import texas_holdem_v1 as texas_holdem
from pettingzoo.classic import texas_holdem_no_limit_v1 as texas_holdem_no_limit
from pettingzoo.classic import uno_v1 as uno
from pettingzoo.classic import dou_dizhu_v1 as dou_dizhu
from pettingzoo.classic import gin_rummy_v1 as gin_rummy
from pettingzoo.classic import go_v1 as go
from pettingzoo.classic import hanabi_v1 as hanabi
from pettingzoo.classic import backgammon_v1 as backgammon

# from pettingzoo.butterfly import knights_archers_zombies_v0 as
# from pettingzoo.butterfly import pistonball_v1
from pettingzoo.butterfly import cooperative_pong_v2 as cooperative_pong
from pettingzoo.butterfly import prison_v2 as prison

from pettingzoo.magent import battle_v2
# from pettingzoo.magent import adversarial_pursuit_v1
# from pettingzoo.magent import gather_v1
# from pettingzoo.magent import combined_arms_v1
# from pettingzoo.magent import tiger_deer_v1
# from pettingzoo.magent import battlefield_v1


from pettingzoo.sisl import pursuit_v3 as pursuit
from pettingzoo.sisl import waterworld_v3 as waterworld
from pettingzoo.sisl import multiwalker_v6 as multiwalker

all_prefixes = ["atari", "classic", "butterfly", "magent", "mpe", "sisl"]

all_environments = {
    "atari/boxing/AEC": boxing,
    # "atari/combat_tank": combat_tank,
    # "atari/combat_plane": combat_plane,
    # "atari/double_dunk": double_dunk,
    # "atari/entombed_cooperative": entombed_cooperative,
    # "atari/entombed_competitive": entombed_competitive,
    # "atari/flag_capture": flag_capture,
    # "atari/joust": joust,
    # "atari/ice_hockey": ice_hockey,
    # "atari/maze_craze": maze_craze,
    "atari/mario_bros/AEC": mario_bros,
    "atari/pong/AEC": pong,
    # "atari/othello": othello,

    "atari/space_invaders/AEC": space_invaders,
    # "atari/space_war": space_war,
    # "atari/surround": surround,
    # "atari/tennis": tennis,
    # "atari/video_checkers": video_checkers,
    # "atari/wizard_of_wor": wizard_of_wor,
    # "atari/warlords": warlords,

    "classic/chess/AEC": chess,
    # "classic/rps": rps,
    # "classic/rpsls": rpsls,
    # "classic/connect_four": connect_four,
    # "classic/tictactoe/AEC": tictactoe,
    # "classic/leduc_holdem": leduc_holdem,
    # "classic/mahjong": mahjong,
    # "classic/texas_holdem": texas_holdem,
    # "classic/texas_holdem_no_limit": texas_holdem_no_limit,
    # "classic/uno": uno,
    # "classic/dou_dizhu": dou_dizhu,
    # "classic/gin_rummy": gin_rummy,
    # "classic/go": go,
    # "classic/hanabi": hanabi,
    # "classic/backgammon": backgammon,


    "butterfly/cooperative_pong/AEC": cooperative_pong,
    "butterfly/prison/AEC": prison,

    # "magent/adversarial_pursuit": adversarial_pursuit_v1,
    "magent/battle/AEC": battle_v2,
    # "magent/battlefield": battlefield_v1,
    # "magent/combined_arms": combined_arms_v1,
    # "magent/gather": gather_v1,
    # "magent/tiger_deer": tiger_deer_v1,


    "sisl/multiwalker/AEC": multiwalker,
    # "sisl/waterworld": waterworld,
    # "sisl/pursuit/AEC": pursuit,
}

render_type =  {
    'magent':"rgb_array",
    'atari':"rgb_array",
    "butterfly":"rgb_array",
    "classic":"stdout",
    "sisl":"rgb_array"
}

PLUGIN_ID = "pistarlab-petting-zoo"
PLUGIN_VERSION = "0.0.1-dev"

from pistarlab.utils.env_helpers import get_env_spec_data
def get_env_specs():
    spec_list = []
    for mod_key, mod in all_environments.items():
        category, game_name, env_type = mod_key.split("/")
        spec = get_env_spec_data(
            spec_id=mod.__name__, 
            env_type=RL_MULTIPLAYER_ENV,
            entry_point="{}:env".format(mod.__name__),
            default_render_mode=render_type.get(category),
            environment_id = mod.__name__,
            categories=[category],
            default_wrappers=[
                {"entry_point":"pistarlab_petting_zoo.wrappers:PettingZooAECWrapper","kwargs":{}}])
        spec_list.append(spec)
    return spec_list

def manifest():
    return {'env_specs': get_env_specs()}

def install():
    ctx.install_plugin_from_manifest(PLUGIN_ID,PLUGIN_VERSION)           
    return True

def load():
    return True

def uninstall():
    ctx.disable_plugin_by_id(PLUGIN_ID)
    return True