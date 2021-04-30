import time


class Timer(object):

    def __init__(self):
        self.timers = {}
        self.timers_log = {}

    def start(self, id):
        self.timers[id] = time.time()
        if id not in self.timers_log:
            self.timers_log[id] = {'counts': 0, 'sums': 0}

    def end(self, id):
        end = time.time()
        diff = end - self.timers[id]
        self.timers_log[id]['counts'] += 1
        self.timers_log[id]['sums'] += diff

    def get_diff(self, id):
        return time.time() - self.timers[id]

    def get_avg(self, id):
        if id not in self.timers_log or self.timers_log[id]['counts'] == 0:
            return 0
        return self.timers_log[id]['sums'] / self.timers_log[id]['counts']
