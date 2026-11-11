import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.num_actions = num_actions
        #camadas conv para extrair features dos frames
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # dimensão apos as conv > 64 canais * 7*7 =3136
        conv_out_size = 64 * 7 * 7
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

    def forward(self, x):
        if x.dim() == 5:
            x = x.squeeze(-1)
        features = self.conv(x)
        features = features.view(features.size(0), -1)
        q_vals = self.fc(features)

        return q_vals