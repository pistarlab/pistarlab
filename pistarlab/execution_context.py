import logging


class RunFuture:

    def __init__(self, task, callback):
        self.task = task
        self.callback = callback

    def get(self):
        import ray
        from ray.exceptions import RayActorError
        try:
            ray.get(self.task)
        except RayActorError:
            pass
        return self.callback()


class ExecutionContext:

    def __init__(self, kwargs, auto_start=False):
        self.kwargs = kwargs
        self.auto_start = auto_start
        import ray
        self.ray = ray
        if not self.ray.is_initialized():
            try:
                logging.info("Initializing Ray with: {}".format(self.kwargs))
                if self.kwargs.get('address') is None:
                    self.kwargs['address'] = "auto"
                self.ray.init(**kwargs)
                logging.info("Connected to Ray")
            except Exception as e:
                if self.auto_start:
                    logging.info("Could not find running Ray instance.  Starting new one.")
                    self.ray.init()
                    logging.info("Ray Initialized")
                else:
                    raise e

    def get(self, task):
        return self.ray.get(task)

    def as_actor(self, clss):
        return self.ray.remote(clss)

    def get_actor_by_uid(self, uid):
        return self.ray.get_actor(name=uid)

    def get_actor_state(self, actor) -> int:
        return self.ray.actors(actor._ray_actor_id.hex()).get("State")

    def get_actor_state_by_uid(self, uid):
        try:
            actor = self.get_actor_by_uid(uid)
            state = self.get_actor_state(actor)
            return state
        except:
            # TODO: document these codes
            return 11

    def stop_actor(self, actor):
        self.ray.kill(actor, no_restart=True)

    def stop_actor_by_uid(self, uid):
        try:
            actor = self.get_actor_by_uid(uid)
            self.stop_actor(actor)
        except:
            pass
