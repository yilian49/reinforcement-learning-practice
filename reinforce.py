from copy import deepcopy

import torch

def reinforce(env, policy, optimizer, episodes=500, gamma=0.99):
    score_history = []
    best_score = 0
    for episode in range(episodes):
        log_probs, rewards, score = policy.test_policy(env)

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
    return score_history, best_model_wts