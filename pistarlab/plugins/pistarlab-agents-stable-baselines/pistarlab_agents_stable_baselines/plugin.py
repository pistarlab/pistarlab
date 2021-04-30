
import logging

PLUGIN_ID = "StableBaselines3"
def install():
    logging.info("Installing: {}".format(PLUGIN_ID))
    from .agent import import_agents
    import_agents(PLUGIN_ID)
    return True


def load():
    import stable_baselines3
    logging.info("Loading: {}".format(PLUGIN_ID))
    return True

def uninstall():
    from pistarlab import ctx
    logging.info("Uninstalling: {}".format(PLUGIN_ID))
    ctx.disable_plugin_by_id(PLUGIN_ID)
    return True