from typing import Union

import torch
from torch import nn
import numpy as np

__all__ = [
    'CartpolePolicy',
    'LSTM',
    'FC',
]

# class CartpolePolicy(nn.Module):
#     def __init__(self):
#         super(CartpolePolicy, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(4,50),
#             nn.ReLU(),
#             nn.Linear(50,50),
#             nn.ReLU(),
#             nn.Linear(50,2),
#             nn.Softmax()
#         )

#     def forward(self, x):
#         return self.model(x)
    
#     def get_action(self,state):
#         state = torch.tensor(state, dtype = torch.float32)
#         action_probs = self.model(state)
#         action = torch.multinomial(action_probs, 1).item()
#         return action

#     def get_hyperparameters(self):
#         return {}

#     def test_policy(self, env, termination = 2000, rounds = 1):
#         log_probs = []
#         rewards = []
#         state = env.reset()[0]
#         counter = 0
#         while True:
#             action = self.get_action(state)
#             next_state, reward, done, _, _ = env.step(action)

#             log_probs.append(torch.log(self.model(torch.tensor(state, dtype=torch.float32))[action]))
#             rewards.append(reward)

#             if done:
#                 break
#             if termination is not None:
#                 if counter > termination:
#                     break
#                 counter += 1
#             state = next_state
#         return log_probs, rewards, sum(rewards)

class CartpolePolicy(nn.Module):
    def __init__(self):
        super(CartpolePolicy, self).__init__()
    
    def test_policy(self, env, termination = 2000):
        log_probs = []
        rewards = []
        state = env.reset()[0]
        counter = 0
        while True:
            action = self.get_action(state)
            next_state, reward, done, _, _ = env.step(action)

            log_probs.append(torch.log(self.model(torch.tensor(state, dtype=torch.float32))[action]))
            rewards.append(reward)

            if done:
                break
            if termination is not None:
                if counter > termination:
                    break
                counter += 1
            state = next_state
        return log_probs, rewards, sum(rewards)
    
    def average_test_policy(self, env, termination = 2000, rounds = 5):
        rewards_total = []
        for r in range(rounds):
            _, _, rewards = self.test_policy(env, termination)
            rewards_total.append(rewards)
        return np.mean(rewards_total), np.std(rewards_total)
    
    def render_env(self, env):
        state = env.reset()[0]
        done = False
        counter = 0
        while not done and counter <=1000:
            env.render()
            action = self.get_action(state)
            state, _, done, _, _ = env.step(action)
            counter += 1
        env.close()
        return counter
    
class LSTM_Cartpole(CartpolePolicy):
    def __init__(self, hidden_dim):
        super(LSTM_Cartpole, self).__init__()
        self.lstm = nn.LSTM(4, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 2)
        self.hidden_dim = hidden_dim
        self.hidden = None

    def get_hyperparameters(self):
        return {
            'hidden_dim' : self.hidden_dim,
        }
    
    def get_action(self,state):
        state = torch.tensor(state, dtype = torch.float32)
        action_probs = self.forward(state)
        action = torch.multinomial(action_probs, 1).item()
        return action

    def reset_hidden(self, batch_size=1):
        self.hidden = (torch.zeros(1, batch_size, self.hidden_dim),
                       torch.zeros(1, batch_size, self.hidden_dim))

    def forward(self, x):
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        output = self.fc(lstm_out)
        return torch.softmax(output, dim=2)
    
    def test_policy(self, env, termination = 2000, rounds = 1):
        self.reset_hidden()
        return super().test_policy(env, termination=termination)
    
class FC_Cartpole(CartpolePolicy):
    def __init__(
            self, 
            topology = [4,50,50,2], 
            activation_function = nn.ReLU,
            dropout = 0.
            ):
        super(FC_Cartpole, self).__init__()

        self.topology = topology
        self.activation_function = activation_function
        self.dropout = dropout
        layers = []
        for i in range(len(topology) - 1):
            layers.append(nn.Linear(topology[i], topology[i+1]))
            if i < len(topology) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Softmax())

        self.model = nn.Sequential(*layers)

    def get_hyperparameters(self):
        return {
            'topology' : self.topology,
            'activation_function' : self.activation_function,
            'dropout' : self.dropout
        }

    def forward(self, x):
        return self.model(x)
    
    def get_action(self,state):
        state = torch.tensor(state, dtype = torch.float32)
        action_probs = self(state)
        action = torch.multinomial(action_probs, 1).item()
        return action