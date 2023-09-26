import os

import gymnasium as gym
import torch
from torch import optim

from models import FC_Cartpole
from reinforce import reinforce
from utils import save_policy_network

exp = 'DomainRandomization_FC_gravity'
if os.path.exists(exp):
    pass
else:
    os.makedirs(exp)

gravities = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14 ,15]

episodes = 1500

def test_gravities(policy):
    env = gym.make('CartPole-v1').unwrapped
    scores = []
    for g in gravities:
        env.gravity = g
        scores.append(policy.average_test_policy(env, rounds = 10, termination = 1000))
    return scores

def main():
    env = gym.make('CartPole-v1')
    policy = FC_Cartpole()
    opt = optim.Adam(policy.parameters(), lr = 0.001)
    score_history, best_params = reinforce(env, policy, opt, episodes=episodes)
    test_scores = test_gravities(policy=policy)
    save_policy_network(
        f'{exp}/baseline.pt', 
        policy_net=policy, 
        training_info={
            'score_history' : score_history,
            'best_params' : best_params,
            'test_scores' : test_scores
        }
        )
    
    policy = FC_Cartpole()
    score_history, best_params = reinforce(env, policy, opt, episodes=episodes, randomize=True)
    test_scores = test_gravities(policy=policy)
    save_policy_network(
        f'{exp}/DR.pt', 
        policy_net=policy, 
        training_info={
            'score_history' : score_history,
            'best_params' : best_params,
            'test_scores' : test_scores
        }
        )
    
if __name__ == '__main__':
    main()