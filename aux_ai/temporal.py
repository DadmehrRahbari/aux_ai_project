# aux_ai_project\aux_ai\temporal.py
from collections import deque
import numpy as np
import params

class ObjectTemporalBuffer:
    def __init__(self, max_len=None):
        self.buffers = {}
        self.max_len = max_len if max_len is not None else params.OBJECT_BUFFER_SIZE

    def update(self, object_id, score):
        if object_id not in self.buffers:
            self.buffers[object_id] = deque(maxlen=self.max_len)
        self.buffers[object_id].append(score)
        return self.get_stability(object_id)

    def get_stability(self, object_id):
        buf = self.buffers[object_id]
        if len(buf) < 2:
            return 1.0 # New objects are assumed stable
        
        std_val = np.std(buf)
        # Stability is high if variance is low
        stability = max(0.0, 1.0 - (std_val * 2)) 
        return float(stability)