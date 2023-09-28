from copy import deepcopy

import torch
import numpy as np

class Reinforce():
    def __init__(
            self,
            length_range = np.random.uniform(0.5, 1.0),
            gravity_range = np.random.uniform(9, 11)
    ):
        self.length_range = length_range
        self.gravity_range = gravity_range

    def set_ranges(self, length_range = None, gravity_range = None):
        if length_range is not None:
            self.length_range = length_range
        if gravity_range is not None:
            self.gravity_range = gravity_range

    def reinforce(
            self,
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
        termination_counter = 0
        for episode in range(episodes):
            if randomize:
                self.randomize_env(env)
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
            if score > termination - 2:
                termination_counter += 1
                if termination_counter > 10:
                    print('terminating early')
                    break
        return score_history, best_model_wts

    def randomize_env(self, env):
        env.length = self.length_range
        env.gravity = self.gravity_range