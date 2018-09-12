class ExperienceBuffer:
    ''' This class stored the agent's memory of playing the games/problems so
        that it can use them to train on. The take_sample function returns a
        random sample of the memories'''

    def __init__(self, max_buffer):
        self._max_memory = max_buffer
        self._samples = []

    def add_sample(self, sample):
        self._samples += [sample]
        if len(self._samples) > self._max_buffer:
            self._samples = self._samples[1:]

    def take_sample(self, no_samples):
        no_samples = min(no_samples, len(self._samples))
        return random.sample(self._samples, no_samples)
