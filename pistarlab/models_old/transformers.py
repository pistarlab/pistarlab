import numpy as np
import gym
from typing import Tuple

def int_to_one_hot_vec(state, size, dtype=np.int):
    data = np.zeros((size), dtype=dtype)
    data[state] = 1.0
    return data


def int_array_to_multi_hot_vec(ints, size, dtype=np.int):
    data = np.zeros((size), dtype=dtype)
    for i in ints:
        data[i] = 1.0
    return data

def default_vec(state, size):
    return state

class Transformer(object):

    def get_shape(self) -> Tuple:
        raise NotImplementedError()

    def fit(self, value):
        raise NotImplementedError()

    def transform(self, value):
        raise NotImplementedError()

    def transform_list(self, values):
        raise NotImplementedError()

class MirrorTransformer(Transformer):
    def __init__(self, size):
        """

        """
        self.size = size
    
    def get_shape(self):
        return self.size

    def fit(self, value):
        return

    def transform(self, value):
        if value.ndim == 1:
            value = np.expand_dims(value,axis=0)
        return value

class ArgMaxOfVecTransformer(Transformer):
    def __init__(self, size):
        """

        """
        self.size = size

    def get_shape(self):
        return self.size

    def fit(self, value):
        return

    def transform(self, value):
        return np.argmax(value)

class NpArrayFlattenTransformer(Transformer):

    def __init__(self, size):
        """

        """
        self.size = size
    
    def get_shape(self):
        return self.size

    def fit(self, value):
        return

    def transform(self, value):
        value = value.flatten()
        if value.ndim == 1:
            value = np.expand_dims(value,axis=0)
        return value

class ImageTransformer(Transformer):

    def __init__(self, shape):
        """

        """
        self.shape = shape
        self.reshape = lambda x:x

        # Model expects first dimension to be channels
        if len(self.shape) == 3 and self.shape[0] > 3:
            self.reshape = lambda x: np.transpose(x,(2,1,0))
            self.shape = (shape[2],shape[1],shape[0])

    
    def get_shape(self):
        return self.shape

    def fit(self, value):
        return

    def transform(self, value):
        return self.reshape(value)


class IntToOneHotTransformer(Transformer):

    def __init__(self, size, dtype=np.int):
        """

        """
        self.size = size
        self.dtype = dtype
    
    def get_shape(self):
        return self.size

    def fit(self, value):
        return

    def transform(self, value):
        value =  int_to_one_hot_vec(value, self.size, self.dtype)
        if value.ndim == 1:
            value = np.expand_dims(value,axis=0)
        return value


def build_observation_transformer(space:gym.Space, for_cnn=False) -> Transformer:
    """
    Uses value to find a transformer

    :return:
    """

    if type(space) == gym.spaces.Discrete:
        if space.n < 1000:
            return IntToOneHotTransformer(space.n)
        else:
            print("Discrete spaces limited to 1000 values")
            raise Exception("Unsupported observation space")
    elif type(space) == gym.spaces.Box:
        print("Box Space Transformer")
        if for_cnn:
            print("...for cnn")
            return ImageTransformer(space.shape)
        else:
            size = np.prod(np.array(list(space.shape)))
            return NpArrayFlattenTransformer(size)
    else:
        raise Exception("Unsupported observation space")


def build_action_transformer(space) -> Transformer:
    if type(space) == gym.spaces.Discrete:
        size = space.n
    elif type(space) == gym.spaces.Box:
        size = np.prod(np.array(list(space.shape)))
    else:
        raise Exception("Unsupported action space")
    return MirrorTransformer(size)


def prep_frames(input_frames, shape, expected_frame_count, transformer):
    # TODO: this can be cleaner
    # print(shape) 
    result = np.zeros(shape)
    frame_array = np.concatenate([transformer.transform(frame) for frame in input_frames])
    if len(shape) ==2:
        result[:frame_array.shape[0],:frame_array.shape[1]] = frame_array
    elif len(shape) ==1:
        result[:frame_array.shape[0]] = frame_array
    else:
        return frame_array
    return np.squeeze(result)