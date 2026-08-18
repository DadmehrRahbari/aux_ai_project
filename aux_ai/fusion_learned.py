# aux_ai_project\aux_ai\fusion_learned.py

import torch
import torch.nn as nn

class LearnedFusion(nn.Module):
    def __init__(self, input_dim=5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16,1),
            nn.Sigmoid()
        )

    def forward(self, scores_dict):
        x = torch.tensor([
            scores_dict.get("ood",0.0),
            scores_dict.get("drift",0.0),
            scores_dict.get("sensor",0.0),
            scores_dict.get("stability",0.0),
            scores_dict.get("agreement",0.0)
        ]).float()
        return self.fc(x)
