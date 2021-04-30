
import logging
from pistarlab import ctx
from pistarlab.meta import RL_MULTIPLAYER_ENV, RL_SINGLEPLAYER_ENV

PLUGIN_ID = "pistarlab-simpleland"
PLUGIN_VERSION = "0.0.1-dev"


from pistarlab.utils.env_helpers import get_env_spec_data
def get_env_specs():
    spec_list = []

    spec = get_env_spec_data(
        spec_id='simpleland_collect', 
        env_type=RL_MULTIPLAYER_ENV,
        entry_point="simpleland.env:SimplelandEnv",
        default_render_mode='rgb_array',
        environment_id = "simpleland",
        categories=[],
        default_wrappers=[])
    spec_list.append(spec)
    print(spec)

    spec2 = get_env_spec_data(
        spec_id='simpleland_collect_single', 
        env_type=RL_SINGLEPLAYER_ENV,
        entry_point="simpleland.env:SimplelandEnvSingle",
        default_render_mode='rgb_array',
        environment_id = "simpleland",
        categories=[],
        default_wrappers=[])
    spec_list.append(spec2)
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