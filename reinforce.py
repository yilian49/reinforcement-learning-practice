from copy import deepcopy

import torch
import numpy as np

def reinforce(
        env, 
        policy, 
        optimizer, 
        episodes=500, 
        gamma=0.99, 
        randomize = False,
        termination = 1000
        ):
    score_history = []
    best_score = 0
    for episode in range(episodes):
        if randomize:
            randomize_env(env)
        log_probs, rewards, score = policy.test_policy(env, termination = termination)

        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        loss = []
        for log_prob, R in zip(log_probs, returns):
            loss.append(-log_prob * R)
        loss = sum(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if episode % 10 == 0:
            print(f"Episode {episode}, Score: {score}")
        score_history.append(score)
        if best_score < score:
            best_score = score
            best_model_wts = deepcopy(policy.state_dict())
            print(f'new best score: {score}')
        if (
            score_history[-1] > termination - 2 and 
            score_history[-2] > termination - 2 and 
            score_history[-3] > termination - 2
        ):
            print('terminating early')
            break
    return score_history, best_model_wts

def randomize_env(env):
    # env.length = np.random.uniform(0.5, 1.0)
    env.gravity = np.random.uniform(8, 11)
