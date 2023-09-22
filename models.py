import torch
from torch import nn

class CartpolePolicy(nn.Module):
    def __init__(self):
        super(CartpolePolicy, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4,50),
            nn.ReLU(),
            nn.Linear(50,50),
            nn.ReLU(),
            nn.Linear(50,2),
            nn.Softmax()
        )
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