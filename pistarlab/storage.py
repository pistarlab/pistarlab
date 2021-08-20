import logging
import os
import shutil
from typing import Any, Dict

import gym
import numpy as np
import simplejson as json

from .filestore import FileStore
from .meta import *

class JSONEncoderCustom(json.JSONEncoder):

    def default(self, obj):  # pylint: disable=E0202
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, gym.Space):
            obj: gym.Space = obj
            return self.encode(obj.__dict__)
        elif isinstance(obj, np.dtype):
            return "dtype:{}".format(obj)
        elif isinstance(obj, np.ndarray):
            if obj.size > 1000:
                return "ENCODE_FAILED:NUMPY_TOO_LARGE,size:{}".format(obj.size)
            else:
                return obj.tolist()
        else:
            try:
                return super().default(obj)
            except Exception as e:
                return "ENCODE_FAILED:UNKNOWN,as_str:{}".format(obj)


class Storage(FileStore):
    """
    TODO: Move FileStore to component (instead of inheritance)
    """

    def __init__(self, root_path, overwrite=False, check_file_last_updated=True, read_only=False, use_write_thread=True):
        super(Storage, self).__init__(root_path,
                                      json_encoder=JSONEncoderCustom,
                                      check_file_last_updated=check_file_last_updated,
                                      read_only=read_only,
                                      use_write_thread=use_write_thread)

        if overwrite:
            logging.info("Overwrite enabled...")
            if os.path.exists(self.root_path):
                logging.info(".... Removing directory {}".format(self.root_path))
                shutil.rmtree(self.root_path)
            else:
                logging.info("No folder exists, not overwriting")

        if not os.path.exists(self.root_path):
            os.makedirs(self.root_path)

    def save_session_data(self, session_id, name, data):
        self.save((SESSION_ENTITY, session_id), name=name, value=data)

    def get_session_data(self,
                         session_id: str,
                         name,
                         start_idx=0,
                         end_idx=None):
        key = (SESSION_ENTITY, session_id)

        return self.get_multipart_dict(key, name, start_idx=start_idx, end_idx=end_idx)

    def save_session_logs(self, session_id, name, logs):
        key = (SESSION_ENTITY, session_id)
        self.extend_multipart_dict(key, name=name, value=logs)



    def save_episode_recording(self,
                               session_id: str,
                               episode_id: str,
                               values):

        key = (SESSION_ENTITY, session_id, EPISODE_ENTITY, episode_id)
        # TODO: For now until better option
        try:
            values['observation'] = [v.shape for v in values['observation']]
        except:
            values['observation'] = ["NA" for v in values['observation']]

        # TODO: split if large
        self.extend_multipart_dict(key, name='recording', value=values)

    def save_episode_stats(self,
                           session_id: str,
                           episode_id: str,
                           values):

        key = (SESSION_ENTITY, session_id, EPISODE_ENTITY, episode_id)
        self.extend_multipart_dict(key, name='stats', value=values)

    def save_image(self, name, image):
        self.save(('meta', 'env_preview'), name=name, value=image, stype='jpg', clear_cache=True)

    def save_frame(self,
                   session_id: str,
                   episode_id: str,
                   step_id: int,
                   frame: Any):

        key = (SESSION_ENTITY, session_id, EPISODE_ENTITY, episode_id)
        self.save(key + ('images',), name=f'{step_id:05}', value=frame, stype='jpg', clear_cache=True)
