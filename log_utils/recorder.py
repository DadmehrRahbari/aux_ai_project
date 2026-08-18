# aux_ai_project/log_utils/recorder.py
import pandas as pd

class Recorder:
    def __init__(self):
        self.logs = []

    def log(self, data: dict):
        """Appends a frame's worth of data (scores, latency, energy for all methods)."""
        self.logs.append(data)

    def export(self, filename="aux_ai_results.csv"):
        df = pd.DataFrame(self.logs)
        # Use CSV for better compatibility with the plotting scripts
        if filename.endswith(".xlsx"):
            df.to_excel(filename, index=False)
        else:
            df.to_csv(filename, index=False)
        return df

    def get_logs(self):
        return self.logs