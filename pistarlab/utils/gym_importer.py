from ..meta import *
from .. import ctx
from gym.envs import registry as gym_registry
import logging
from .env_helpers import get_env_spec_data

# TODO Move to plugin
ENV_EXCLUDE_SET = {'Defender-v0', 'Defender-v4',
                   'DefenderDeterministic-v0', 'DefenderDeterministic-v4', 'DefenderNoFrameskip-v0', 'DefenderNoFrameskip-v4', 'Defender-ram-v0',
                   'Defender-ram-v4', 'Defender-ramDeterministic-v0', 'Defender-ramDeterministic-v4', 'Defender-ramNoFrameskip-v0', 'Defender-ramNoFrameskip-v4'}

ATARI_PREFIX_MATCHER = ['JourneyEscape', 'Solaris', 'Gravitar', 'Adventure', 'Acrobot', 'AirRaid', 'Alien', "Carnival", "Berzerk", "BipedalWalker",
                        "MontezumaRevenge", "FrozenLake", "CrazyClimber", "ChopperCommand", "Centipede", "BeamRider", "Tutankham",
                        'Amidar', 'Assault', 'Asterix', 'Asteroids', 'Atlantis', "MsPacman", "MountainCar", "WizardOfWor",
                        "Hero", "DemonAttack", "CubeCrash", 'BankHeist', 'BattleZone', 'Breakout', 'Boxing', "YarsRevenge",
                        'Bowling', "NameThisGame", "Enduro", "DoubleDunk", "Tennis", "TimePilot", "StarGunner", "FishingDerby", "Venture", "UpNDown",
                        "SpaceInvaders", "Seaquest", "Robotank", "ElevatorAction", "Gopher", "Frostbite", "Freeway", "VideoPinball",
                        "ReversedAddition", "PrivateEye", "Pooyan", "Pong", "Pitfall", "Phoenix", "Jamesbond", "IceHockey", "Skiing",
                        'Qbert', "Riverraid", "RoadRunner", "Zaxxon", "LunarLander", "KungFuMaster", "Krull", "Kangaroo"]


# RENDER_OVERRIDE = {
#     'CliffWalking-v0': 'human'
# }

def get_env_specs_from_gym_registry(
        entry_point_prefix,
        environment_id_filter_set=None,
        default_wrappers=[],
        max_count=None,
        gym_prefix_group_matcher=ATARI_PREFIX_MATCHER,
        env_exclude_set=ENV_EXCLUDE_SET,
        default_render_mode=None,
        env_type=RL_SINGLEPLAYER_ENV,
        additional_tags=[],
        additional_categories=[],
        version="0.0.1-dev"):

    counter = 0
    spec_list = []
    for gym_spec in gym_registry.all():
        if (gym_spec.id not in env_exclude_set) and \
            (gym_spec.entry_point is not None) and \
                (entry_point_prefix in gym_spec.entry_point.lower()):

            logging.info("Adding specID {}".format(gym_spec.id))

            # get group id
            parts = gym_spec.id.split("-", 1)
            environment_id = parts[0]
            for prefix in gym_prefix_group_matcher:
                if environment_id.startswith(prefix):
                    environment_id = prefix

            if environment_id_filter_set is not None and environment_id not in environment_id_filter_set:
                continue

            spec = get_env_spec_data(spec_id=gym_spec.id,
                                     entry_point=gym_spec.entry_point,
                                     env_kwargs=gym_spec._kwargs,
                                     env_type=env_type,
                                     version=version,
                                     tags=additional_tags,
                                     categories=additional_categories,
                                     environment_id=environment_id,
                                     default_wrappers=default_wrappers,
                                     default_render_mode=default_render_mode)
            spec_list.append(spec)

            counter += 1

            if max_count and counter >= max_count:
                logging.info("Max count reached, skipping remainder of envs")
                break
        else:
            logging.debug("Skipping Env: {} {}".format(gym_spec.id, gym_spec.entry_point))

    logging.info("Added {} new envs".format(counter))
    return spec_list
