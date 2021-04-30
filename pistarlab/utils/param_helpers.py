import collections
import json

def flatten(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def create_params_from_flattened(flattened_config,overrides={},interfaces=[],mode="advanced"):
    params = {}
    for k, v in flattened_config.items():
        if v is not None:
            tp = type(v)
            if tp is list:
                v = json.dumps(v)

            params[k] = {
                'displayed_name': k.split('.')[-1],
                "description": "",
                'path': k,
                "default": v,
                "mode":mode,
                "data_type": tp.__name__,
                "type_info": {},
                "interfaces":interfaces}

            params[k].update(overrides)
    return params


def create_params_from_dict(config, prefix, overrides={},interfaces=[],mode="advanced"):
    flattened_config = flatten(config, prefix)
    return create_params_from_flattened(flattened_config,overrides=overrides,interfaces=interfaces,mode=mode)
