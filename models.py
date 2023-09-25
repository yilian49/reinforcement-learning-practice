from typing import Union

import torch
from torch import nn

class CartpolePolicy(nn.Module):
    def __init__(self, model : Union[None, nn.Module] = None):
        super(CartpolePolicy, self).__init__()
        if model is None:
            self.model = FC()
        else:
            self.model = model
    def forward(self, x):
        return self.model(x)
    
    def get_action(self,state):
        state = torch.tensor(state, dtype = torch.float32)
        action_probs = self.model(state)
        action = torch.multinomial(action_probs, 1).item()
        return action

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
    
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden = None

    def get_hyperparameters(self):
        return {
            'input_dim' : self.input_dim,
            'hidden_dim' : self.hidden_dim,
            'output_dim' : self.output_dim,
        }

    def reset_hidden(self, batch_size=1):
        self.hidden = (torch.zeros(1, batch_size, self.hidden_dim),
                       torch.zeros(1, batch_size, self.hidden_dim))

    def forward(self, x):
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        output = self.fc(lstm_out)
        return torch.softmax(output, dim=2)
    
class FC(nn.Module):
    def __init__(
            self, 
            topology = [4,50,50,2], 
            activation_function = nn.ReLU,
            dropout = 0.
            ):
        super(FC, self).__init__()

        self.topology = topology
        self.activation_function = activation_function
        self.dropout = dropout
        layers = []
        for i in range(len(topology) - 1):
            layers.append(nn.Linear(topology[i], topology[i+1]))
            if i < len(topology) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def get_hyperparameters(self):
        return {
            'topology' : self.topology,
            'activation_function' : self.activation_function,
            'dropout' : self.dropout
        }

    def forward(self, x):
        return self.model(x)