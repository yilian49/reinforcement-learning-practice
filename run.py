import os

import gymnasium as gym
import torch
from torch import optim

from models import FC_Cartpole
from reinforce import reinforce
from utils import save_policy_network

exp = 'DomainRandomization_FC_gravity_length'
if os.path.exists(exp):
    pass
else:
    os.makedirs(exp)

gravities = [ 7, 9, 9.8,  10, 12, 14 ,16, 18, 20]
lengths = [0.3, 0.4, 0.5, 0.7, 0.9, 1.1, 1.5, 2.0, 2.5, 3.0]
episodes = 1500

def test_gravities(policy):
    env = gym.make('CartPole-v1').unwrapped
    scores = []
    for g in gravities:
        env.gravity = g
        scores.append(policy.average_test_policy(env, rounds = 10, termination = 1000))
    return scores

def test_gravity_and_lengths(policy):
    env = gym.make('CartPole-v1').unwrapped
    scores = []
    for g in gravities:
        scores.append([])
        for l in lengths:
            env.gravity = g
            env.length = l
            scores[-1].append(policy.average_test_policy(env, rounds = 10, termination = 1000))
    return scores

def main():
    env = gym.make('CartPole-v1')
    policy = FC_Cartpole(topology=[4,100,100,100,2])
    opt = optim.Adam(policy.parameters(), lr = 0.001)
    # policy = FC_Cartpole()
    # opt = optim.Adam(policy.parameters(), lr = 0.001)
    score_history, best_params = reinforce(env, policy, opt, episodes=episodes)
    test_scores = test_gravity_and_lengths(policy=policy)

    save_policy_network(
        f'{exp}/baseline.pt', 
        policy_net=policy, 
        training_info={
            'score_history' : score_history,
            'best_params' : best_params,
            'test_scores' : test_scores,
            'gravities' : gravities,
            'lengths' : lengths
        }
        )
    
    policy = FC_Cartpole(topology=[4,100,100,100,2])
    opt = optim.Adam(policy.parameters(), lr = 0.001)
    score_history, best_params = reinforce(env, policy, opt, episodes=episodes, randomize=True)
    test_scores = test_gravity_and_lengths(policy=policy)
    save_policy_network(
        f'{exp}/DR.pt', 
        policy_net=policy, 
        training_info={
            'score_history' : score_history,
            'best_params' : best_params,
            'test_scores' : test_scores,
            'gravities' : gravities,
            'lengths' : lengths
        }
        )
    
if __name__ == '__main__':
    main()