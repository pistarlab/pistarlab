import numpy as np



# source https://github.com/openai/gym-http-api/blob/master/gym_http_server.py
def get_space_properties(space):
    info = {}
    info['name'] = space.__class__.__name__
    if info['name'] == 'Discrete':
        info['n'] = space.n
    elif info['name'] == 'Box':
        info['shape'] = space.shape
        # It's not JSON compliant to have Infinity, -Infinity, NaN.
        # Many newer JSON parsers allow it, but many don't. Notably python json
        # module can read and write such floats. So we only here fix "export version",
        # also make it flat.
        info['low'] = [(x if x != -np.inf else -1e100) for x in np.array(space.low).flatten()]
        info['high'] = [(x if x != +np.inf else +1e100) for x in np.array(space.high).flatten()]
    elif info['name'] == 'HighLow':
        info['num_rows'] = space.num_rows
        info['matrix'] = [((float(x) if x != -np.inf else -1e100) if x != +np.inf else +1e100) for x in np.array(space.matrix).flatten()]
    else:
        info['message'] = "Serialization not implemented"
    return info


def space_to_pyson(space):
    """
    converts a space into json and valid pyson

    reference: https://github.com/openai/gym-http-api/blob/master/gym_http_server.py
    """

    name = space.__class__.__name__
    module = space.__class__.__module__
    pydef = {}
    pydef['class_name'] = name
    pydef['module'] = module

    args = []
    kwargs = {}
    if name == 'Discrete':
        args.append(int(space.n))
    elif name == 'Box':
        kwargs['shape'] = {
            "__type": "tuple",
            "values": [int(x) for x in list(space.shape)]
        }
        # It's not JSON compliant to have Infinity, -Infinity, NaN.
        # Many newer JSON parsers allow it, but many don't. Notably python json
        # module can read and write such floats. So we only here fix "export version",
        # also make it flat.
        kwargs['low'] = float(np.min([(x if x != -np.inf else -1e100) for x in np.array(space.low).flatten()]))
        kwargs['high'] = float(np.max([(x if x != +np.inf else +1e100) for x in np.array(space.high).flatten()]))
    elif name == 'HighLow':
        kwargs['num_rows'] = int(space.num_rows)
        kwargs['matrix'] = [((float(x) if x != -np.inf else -1e100) if x != +np.inf else +1e100) for x in np.array(space.matrix).flatten()]
    elif name.lower() == "tuple" and module == "gym.spaces.tuple":
        kwargs['spaces'] = [space_to_pyson(s) for s in space.spaces]
    elif name.lower() == "dict":
        kwargs['spaces'] = {n: space_to_pyson(s) for n, s in space.spaces.items()}
    elif name.lower() == "DataSpace":
        kwargs['includes_targets'] = "TODO"
        kwargs['spaces'] = "TODO"  # [space_to_pyson(s) for s in space.spaces]
    else:
        raise NotImplementedError("Py def not implemented for class:{}, module:{}, object:{}".format(name, module, space))
    pydef['kwargs'] = kwargs
    pydef['args'] = args

    return pydef
