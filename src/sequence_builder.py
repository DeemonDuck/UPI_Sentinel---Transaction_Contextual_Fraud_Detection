import numpy as np

class SequenceBuilder:
    def __init__(self, sequence_length):
        
        self.sequence_length = sequence_length
        self.buffer = []

    def update(self, txn):

        self.buffer.append(txn[0])

        if len(self.buffer) > self.sequence_length:
            self.buffer.pop(0)

        return np.array(self.buffer)

    def is_ready(self):
        return len(self.buffer) == self.sequence_length

    def reset(self):
        self.buffer = []