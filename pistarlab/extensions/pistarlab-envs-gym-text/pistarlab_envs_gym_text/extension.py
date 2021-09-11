import logging
from pistarlab import ctx

from pistarlab.extension_tools import load_extension_meta
EXT_META = load_extension_meta(__name__)
EXTENSION_ID = EXT_META["id"]
EXTENSION_VERSION =  EXT_META["version"]

from pistarlab.utils.gym_importer import get_environments_from_gym_registry

def manifest():
    envs_all = []
    
    envs = get_environments_from_gym_registry(
        entry_point_prefix=f"gym.envs.toy_text",
        additional_categories=['toy_text'],
        collection="Gym Text",
        env_description="OpenAI Gym Toy Text Environment",
        env_usage="See: https://gym.openai.com/envs/#toy_text")
    envs_all.extend(envs)
    envs = get_environments_from_gym_registry(
        entry_point_prefix=f"gym.envs.unittest",
        additional_categories=['unittest'],
        collection="Gym Tests",
        env_description="OpenAI Gym Unit Test Environment",
        env_usage="See: https://gym.openai.com/envs/")
    envs_all.extend(envs)
    
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