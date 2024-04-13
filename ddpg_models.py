import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DDPG_Actor(nn.Module):
    def __init__(self, in_channels=1, num_actions=3):
        super(DDPG_Actor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 84, kernel_size=4, stride=4)
        self.conv2 = nn.Conv2d(84, 42, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(42, 21, kernel_size=2, stride=2)
        self.fc4 = nn.Linear(21*4*4, 168)
        self.fc5 = nn.Linear(168, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc4(x))
        return torch.tanh(self.fc5(x))

class DDPG_Critic(nn.Module):
        def __init__(self):
            super(DDPG_Critic, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=84, kernel_size=4, stride=4)
            self.conv2 = nn.Conv2d(84, 42, kernel_size=4, stride=2)
            self.conv3 = nn.Conv2d(42, 21, kernel_size=2, stride=2)
            self.fc4 = nn.Linear(21*4*4, 168)

            self.fc5 = nn.Linear(3, 168)
            self.fc6 = nn.Linear(168*2, 84)
            self.fc7 = nn.Linear(84, 3)           

        def forward(self, state, action):
            x = F.relu(self.conv1(state))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc4(x))
            action = F.relu(self.fc5(action))
            x = torch.cat([x, action], dim=1)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc6(x))
            return self.fc7(x)
            
class OrnsteinUhlenbeckActionNoise():
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2, dt=1e-2):
        self.action_dim = action_dim
        self.mu = mu
        self.dt = dt
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X) * self.dt
        dx = dx + self.sigma * np.sqrt(self.dt) * np.random.randn(len(self.X))
        self.X = self.X + dx
        return torch.tensor(self.X)
    
