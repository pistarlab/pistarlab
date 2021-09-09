from ..meta import *
from .. import ctx
from gym.envs import registry as gym_registry
import logging
from .env_helpers import get_environment_data, get_env_spec_data
from gym import make
# TODO Move to extension
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

# Wrapps call to gym make function so we can use 'id' as a kwargs
def gym_make(id=None,**kwargs):
    if id is None:
        raise Exception("call to gym_make requires id but is None")

    return make(id,**kwargs)

def get_environments_from_gym_registry(
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
        collection = None,
        version="0.0.1-dev",
        force_environment_id=None,
        force_environment_displayed_name=None):

    counter = 0
    envs = {}
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

            if force_environment_id is not None:
                environment_id = force_environment_id
                env_displayed_name = force_environment_displayed_name or environment_id.title().replace("_", " ")
            else:
                env_displayed_name = environment_id.title().replace("_", " ")
            
            env = envs.get(environment_id)
            if env is None:
                env = get_environment_data(
                    environment_id=environment_id,
                    displayed_name=env_displayed_name,
                    categories=additional_categories,
                    collection = collection,
                    version=version,
                    env_specs=[])

            spec = get_env_spec_data(spec_id=gym_spec.id,
                                     entry_point="pistarlab.utils.gym_importer:gym_make",
                                     env_kwargs={'id':gym_spec.id},
                                     env_type=env_type,
                                     tags=additional_tags,
                                     default_wrappers=default_wrappers,
                                     default_render_mode=default_render_mode)
            env['env_specs'].append(spec)
            envs[environment_id] = env

            counter += 1

            if max_count and counter >= max_count:
                logging.info("Max count reached, skipping remainder of envs")
                break
        else:
            logging.debug("Skipping Env: {} {}".format(gym_spec.id, gym_spec.entry_point))

    logging.info("Added {} new envs".format(counter))
    return list(envs.values())
