import logging
from pistarlab import ctx

from pistarlab.extension_tools import load_extension_meta
EXT_META = load_extension_meta(__name__)
EXTENSION_ID = EXT_META["id"]
EXTENSION_VERSION =  EXT_META["version"]

from pistarlab.utils.gym_importer import get_environments_from_gym_registry

def manifest():
    envs_all = []
    classic_control_envs = get_environments_from_gym_registry(
        entry_point_prefix=f"gym.envs.classic_control",
        additional_categories=['classic_control'],
        collection="Gym Classic Control",
        env_description="OpenAI Gym Control theory problem",
        env_usage="See: https://gym.openai.com/envs/#classic_control"
    )
    envs_all.extend(classic_control_envs)

    box2d_envs = get_environments_from_gym_registry(
        entry_point_prefix=f"gym.envs.box2d",
        additional_categories=['box2d'],
        collection="Gym Box2d",
        env_description="OpenAI Gym Continuous control task in the Box2D simulator.",
        env_usage="See: https://gym.openai.com/envs/#box2d"
    )

    envs_all.extend(box2d_envs)

    return {'environments': envs_all}

def install():
    ctx.install_extension_from_manifest(EXTENSION_ID,EXTENSION_VERSION)
    return True

def load():
    return True

def uninstall():
    logging.info("Uninstalling {}".format(EXTENSION_ID))
    ctx.disable_extension_by_id(EXTENSION_ID)
    return True