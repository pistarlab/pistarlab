from typing import Dict, Any


class SessionConfig:

    def __init__(self, tmp_session=False):
        self.tmp_session = tmp_session

    def get_dict(self) -> Dict[str, Any]:
        kwargs = {}
        kwargs.update(self.__dict__)
        return kwargs


class DataSessionConfig(SessionConfig):

    def __init__(self,
                 batch_size=50,
                 batch_log_freq=100,
                 max_epochs=None,
                 max_batches=None,
                 wrappers=[],
                 tmp_session=False):
        super().__init__(tmp_session=tmp_session)
        self.batch_size = batch_size
        self.batch_log_freq = batch_log_freq
        self.max_epochs = max_epochs
        self.max_batches = max_batches
        self.wrappers = wrappers


class RLSessionConfig(SessionConfig):

    def __init__(
        self,
        max_episodes=None,
        max_steps=None,
        max_steps_in_episode=None,
        episode_record_freq=1000,
        episode_record_preview_interval=1,
        step_log_freq=100,
        episode_log_freq=1,
        preview_rendering=True,
        frame_stream_enabled=False,
        wrappers=[],
        meta={},
        tmp_session=False,
        inference_only = False
    ):
        super().__init__(tmp_session=tmp_session)
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.max_steps_in_episode = max_steps_in_episode
        self.episode_record_freq = episode_record_freq
        self.step_log_freq = step_log_freq
        self.episode_record_preview_interval = episode_record_preview_interval
        self.episode_log_freq = episode_log_freq
        self.preview_rendering = preview_rendering
        self.frame_stream_enabled = frame_stream_enabled
        self.meta = meta
        self.wrappers = wrappers
        self.inference_only = inference_only
