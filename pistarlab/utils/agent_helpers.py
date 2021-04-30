import logging


def get_agent_spec_interface_dict(
        interface_id='run',  # Same as function name in runner
        interface_type='rl',
        observation_space=None,
        action_space=None,
        auto_config_spaces=True):
    return {
        'interface_id': interface_id,
        'interface_type': interface_type,
        'auto_config_spaces': auto_config_spaces,
        'observation_space': observation_space,
        'action_space': action_space
    }


def get_agent_spec_dict(
        spec_id,
        runner_entry_point,
        entry_point=None,
        config={},
        components={},
        interfaces=None,
        params={},
        disabled=False,
        displayed_name=None,
        version="0.0.1-dev",
        description=None):

    spec_data = {}
    spec_data['spec_id'] = spec_id
    spec_data['displayed_name'] = displayed_name or spec_id
    spec_data['description'] = description
    spec_data['entry_point'] = entry_point
    spec_data['runner_entry_point'] = runner_entry_point
    spec_data['version'] = version
    spec_data['disabled'] = disabled
    spec_data['config'] = config
    spec_data['config']['interfaces'] = interfaces or {'run': get_agent_spec_interface_dict()}
    spec_data['config']['components'] = components
    spec_data['params'] = params

    return spec_data

# default_agent_spec_list = []
# default_agent_spec_list.append({
#     'id':'reinforcev1',
#     'displayed_name': "REINFORCE Basic",
#     'class_name': 'ReinforceAgent',
#     'module': 'pistarlab.agents.reinforce_agent',
#     'description': 'Description HERE',
#     'categories': ['policy_gradient'],
#     'kwarg_spec': {},
#     'kwarg_defaults': {},
#     'displayed_order': 0,

# })

# default_agent_spec_list.append({
#     'id':'reinfocev2',
#     'displayed_name': "REINFORCE V2",
#     'class_name': 'ReinforceV2Agent',
#     'module': 'pistarlab.agents.reinforce_agent_v2',
#     'description': 'TODO',
#     'categories': ['policy_gradient'],
#     'kwarg_spec': {},
#     'kwarg_defaults': {},
#     'displayed_order': 1,
# })

# default_agent_spec_list.append({
#     'id':'crossentropy',
#     'displayed_name': "Cross Entropy",
#     'class_name': 'CrossEntropyAgent',
#     'module': 'pistarlab.agents.cross_entropy_v1',
#     'description': 'TODO',
#     'categories': [],
#     'kwarg_spec': {},
#     'kwarg_defaults': {},
#     'displayed_order': 2,
# })

# default_agent_spec_list.append({
#     'id':'random',
#     'displayed_name': "Random",
#     'class_name': 'RandomAgent',
#     'module': 'pistarlab.agents.random',
#     'description': 'Description HERE',
#     'categories': [],
#     'kwarg_spec': {},
#     'kwarg_defaults': {},
#     'displayed_order': 3
# })

# default_agent_spec_list.append({
#     'id':'webhuman1',
#     'displayed_name': "Human Over Browser",
#     'class_name': 'HumanWeb',
#     'module': 'pistarlab.agents.humanweb',
#     'description': 'NOT IMPLMENTED YET - Browser Interface for human environment interaction',
#     # 'disabled': True,
#     'categories': [],
#     'kwarg_spec': {},
#     'kwarg_defaults': {},
#     'displayed_order': 4
# })

# variables_from_env_def = {
#     "action_space": {
#         "__type": "var",
#         "name": "env.action_space"
#     },
#     "observation_space": {
#         "__type": "var",
#         "name": "env.observation_space"
#     }}
