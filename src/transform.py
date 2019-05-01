import numpy as np
import math

class Transform:
    def __init__(self, bars=1, events=4, note_count=60):
        self.split_size = bars*events
        self.note_count = note_count
        
    def get_sections(self, sample_length):
        return math.ceil(sample_length / self.split_size)
    
    def __call__(self, sample, section):
        start = section*self.split_size
        end = min(section*self.split_size + self.split_size, sample.shape[0])
        sample = sample[start : end]
        
        sample_length = len(sample)
        leftover = sample_length % self.split_size
        if leftover != 0 :
            padding_size = self.split_size - leftover
            padding = np.zeros((padding_size, sample.shape[1], sample.shape[2]), dtype=float)
            sample = np.concatenate((sample, padding), axis=0)
        return sample