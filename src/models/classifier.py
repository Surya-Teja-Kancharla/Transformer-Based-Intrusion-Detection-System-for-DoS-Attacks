import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, d_model, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model//2, num_classes)
        )

    def forward(self, x):
        return self.net(x)
