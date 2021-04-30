class PModel(object):
    def save(self, prefix, store):
        raise NotImplementedError

    def load(self, prefix, store):
        raise NotImplementedError


class PolicyEstimator(PModel):

    def predict_probs(self, state):
        raise NotImplementedError

    def predict_choice(self, state):
        raise NotImplementedError

    def update(self, state, target, action):
        raise NotImplementedError


class ValueEstimator(PModel):

    def predict(self, state):
        raise NotImplementedError

    def update(self, state, target):
        raise NotImplementedError
