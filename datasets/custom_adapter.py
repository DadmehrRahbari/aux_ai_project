# aux_ai_project\datasets\custom_adapter.py
from datasets.base import DatasetAdapter

class CustomDatasetAdapter(DatasetAdapter):
    def __init__(self, name, train=False, root="./data"):
        # Load CSV, tabular, or sensor data here
        self.data = []

    def __iter__(self):
        for sample in self.data:
            yield sample

    def __len__(self):
        return len(self.data)
