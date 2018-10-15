import random


class ExperienceBuffer:
    ''' This class stored the agent's memory of playing the games/problems so
        that it can use them to train on. The take_sample function returns a
        random sample of the memories'''

    def __init__(self, max_memory):
        self.max_memory = max_memory
        self.samples = []

    def add_sample(self, sample):
        self.samples += [sample]
        if len(self.samples) > self.max_memory:
            self.samples = self.samples[1:]

    def take_sample(self, no_samples):
        no_samples = min(no_samples, len(self.samples))
        return random.sample(self.samples, no_samples)
