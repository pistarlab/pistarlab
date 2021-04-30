from pistarlab.dbmodels import *
import logging
from .agent import AGENT_REG

from pistarlab.utils.param_helpers import create_params_from_dict

PLUGIN_ID = "pistarlab-rllib"
PLUGIN_VERSION = "0.0.1-dev"


def get_task_specs():

    spec = {
        'spec_id': 'rllib_multiagent',
        'displayed_name': "Multi-Agent Session for RLlib Agents",
        'description': 'Custom task for running multi agent sessions in rllib ',
        'entry_point': 'pistarlab.task:Task',
        'runner_entry_point': 'pistarlab_rllib.rllib_multiagent_task:RLLibMultiAgentRunner',
        'version': "0.0.1-dev",
        'config': {
            'agent_data_map': [],
            'agent_assignments': {},
            'env_spec_id': 'REQUIRED',
            'env_kwargs': {},
            'session_config': {}
        }
    }

    return [spec]

from pistarlab.utils.agent_helpers import get_agent_spec_dict,get_agent_spec_interface_dict

def get_agent_specs():
    spec_list = []
    for policy_name, reg_fn in AGENT_REG.items():
        data = reg_fn()
        logging.info("LOADING")
        logging.info(policy_name)

        agent_spec = get_agent_spec_dict(
            spec_id = 'rllib_{}'.format(policy_name),
            entry_point = 'pistarlab_rllib.agent:RLlibAgent',
            runner_entry_point = 'pistarlab_rllib.agent:RLlibAgentRunner',
            config={
                'policy_name': policy_name,
                "trainer_config": data['trainer_config']
            },
            components=data.get('components'),
            interfaces={'run':get_agent_spec_interface_dict()},
            params=data.get("params"),
            disabled=False,
            displayed_name="{} - RLlib".format(policy_name),
            version="0.0.1-dev",
            description='RLlib: https://docs.ray.io/en/master/rllib.html\n\n{}'.format("TODO..."))

        spec_list.append(agent_spec)
    return spec_list


def get_component_specs():
    spec_list = []
    spec_list.append({
        'spec_id': 'rllib_fcnet_torch',
        'entry_point': 'ray.rllib.models.torch.fcnet:FullyConnectedNetwork',
        'parent_class_entry_point': 'ray.rllib.models.modelv2:ModelV2',
        'description': "Torch Fully Connected",
        'version': "0.0.1-dev",
        'config': {'model_config': {}},
        'category': "model"})

    spec_list.append({
        'spec_id': 'rllib_fcnet_tf',
        'entry_point': 'ray.rllib.models.tf.fcnet:FullyConnectedNetwork',
        'parent_class_entry_point': 'ray.rllib.models.modelv2:ModelV2',
        'description': "Tensorflow Fully Connected",
        'version': "0.0.1-dev",
        'config': {'model_config': {}},
        'category': "model"})

    spec_list.append({
        'spec_id': 'rllib_fcnet_tf',
        'entry_point': 'ray.rllib.models.tf.fcnet:FullyConnectedNetwork',
        'parent_class_entry_point': 'ray.rllib.models.modelv2:ModelV2',
        'description': "Tensorflow Fully Connected",
        'version': "0.0.1-dev",
        'config': {'model_config': {}},
        'category': "model"})

    spec_list.append({
        'spec_id': 'rllib_visionnet_tf',
        'entry_point': 'ray.rllib.models.tf.visionnet:VisionNetwork',
        'parent_class_entry_point': 'ray.rllib.models.modelv2:ModelV2',
        'description': "Tensorflow Vision (via Convolutions) Model",
        'version': "0.0.1-dev",
        'config': {'model_config': {}},
        'category': "model"})

    spec_list.append({
        'spec_id': 'rllib_visionnet_torch',
        'entry_point': 'pistarlab_rllib.default_vision_model:VisionModel',
        'parent_class_entry_point': 'ray.rllib.models.modelv2:ModelV2',
        'description': "Torch Vision (via Convolutions) Model",
        'config': {'model_config': {}},
        'category': "model"})

    spec_list.append({
        'spec_id': 'rllib_visionnet_torch_custom',
        'entry_point': 'pistarlab_rllib.custom_vision_model:VisionModel',
        'parent_class_entry_point': 'ray.rllib.models.modelv2:ModelV2',
        'description': "Customized Torch Vision (via Convolutions) Model",
        'version': "0.0.1-dev",
        'config': {
            'model_config': {
                'conv_filters': [[16, [4, 4], 2],
                                 [32, [4, 4], 2],
                                 [256, [11, 11], 1], ],
                'num_outputs': 20
            }
        },
        'category': "model"})

    spec_list.append({
        'spec_id': 'rllib_gtrxlnet_torch',
        'entry_point': 'ray.rllib.models.torch.attention_net:GTrXLNet',
        'parent_class_entry_point': 'ray.rllib.models.modelv2:ModelV2',
        'description': "Torch Attention Model",
        'version': "0.0.1-dev",
        'config': {
            'model_config': {
                "num_transformer_units": 1,
                "attn_dim": 64,
                "num_heads": 2,
                "memory_tau": 50,
                "head_dim": 32,
                "ff_hidden_dim": 32}},
        'category': "model"})

    # add model_config params
    for spec in spec_list:
        params = create_params_from_dict(
            spec['config']['model_config'], 
            'model_config', 
            overrides={'mode': "default"})
        spec['params'] = params

    return spec_list


def get_spec_registry_data():
    return {
        'components': get_component_specs(),
        'agents': get_agent_specs(),
        'tasks': get_task_specs()}


def install():
    from pistarlab import ctx
    for spec in get_agent_specs():
        ctx.register_agent_spec(**spec, plugin_id=PLUGIN_ID, plugin_version=PLUGIN_VERSION)

    for spec in get_component_specs():
        ctx.register_component_spec(**spec, plugin_id=PLUGIN_ID, plugin_version=PLUGIN_VERSION)

    for spec in get_task_specs():
        ctx.register_task_spec(**spec, plugin_id=PLUGIN_ID, plugin_version=PLUGIN_VERSION)

    return True


def load():
    from pistarlab import ctx
    from ray.rllib.models import ModelCatalog
    from pistarlab.util_funcs import get_class_from_entry_point

    # Add model to registery at startup
    for spec in get_component_specs():
        spec_id = spec['spec_id']
        cls = get_class_from_entry_point(ctx.get_component_spec(spec_id).entry_point)
        ModelCatalog.register_custom_model(spec_id, cls)

    return True


def uninstall():
    from pistarlab import ctx
    ctx.disable_plugin_by_id(PLUGIN_ID)
    return True


def create_spec_list_file():
    """
    Generate Plugin Meta data file
    """
    import sys
    sys.path.append("../")
    import json
    with open("{}_v{}.json".format(PLUGIN_ID, PLUGIN_VERSION), 'w') as f:
        data = get_spec_registry_data()
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    """
    Instructions: From the plugin root package directory run: 
        python -m  pistarlab_rllib.plugin
    """
    create_spec_list_file()
