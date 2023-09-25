import gymnasium as gym
import torch
from torch import optim

from models import CartpolePolicy
from reinforce import reinforce

def main():
    env = gym.make('CartPole-v1')
    policy = CartpolePolicy()
    opt = optim.Adam(policy.parameters(), lr = 0.001)
    score_history, best_params = reinforce(env, policy, opt, episodes=500, randomize=True)
    torch.save({
        'score_history' : score_history,
        'best_policy_weights' : best_params,
        'policy_weights' : policy.state_dict()
    }, 'reinforce_results_randomize.pt')

if __name__ == '__main__':
    main()