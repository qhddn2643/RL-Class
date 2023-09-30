import numpy as np
import matplotlib.pyplot as plt
import itertools
from collections import defaultdict
from IPython import display

class Actor_Critic_Eligibility_Traces:
    def __init__(self, env, n_episode, lamda, alpha, gamma) -> None:
        
        self.env = env
        self.n_episode = n_episode
        self.lamda = lamda
        self.alpha = alpha
        self.gamma = gamma
        
    def fourier_basis(self, state):
        n = self.env.action_space.n
        X = []
        if type(state) == "<class 'int'>":
            k = state
            X.append([np.cos(np.pi * S * i) for i in range((n+1)**k)])
        if type(state) == "<class 'tuple'>":
            S = np.asarray(state)
            k = len(S)
            X.append([np.cos(np.pi * S.T * i) for i in range((n+1)**k)])
        X = np.array(X)
        return X

    def policy(self, state_action):
        x = np.maximum(0, state_action)
        action_probs = np.exp(x) / np.sum(np.exp(x), axis=0) # softmax
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs) # choice the calculated probability vector to select the next action
        return action

    def V(self, S, w):
        V = w.T * self.fourier_basis(S)
        return V
    
    def dln(self, pi, S, A):
        total = 0
        x = pi[S][A]
        total += (np.exp(x) / np.sum(np.exp(x), axis=0)) * x
        return total


    def play(self):
        pi = defaultdict(lambda: np.zeros(self.env.action_space.n))
        num_step = np.zeros(self.n_episode)
        sum_reward = np.zeros(self.n_episode)
        w = np.zeros(len(self.fourier_basis(self.env.observation_space.shape[0])))
        theta = np.zeros(len(self.fourier_basis(self.env.observation_space.shape[0])))
        score = 0

        for episode in range(self.n_episode):

            obs = self.env.reset()
            S = obs[0], obs[1], obs[2], obs[3], obs[4], obs[5]
            
            t = 1
            z_theta = 0
            z_w = 0
            I = 1
            
            terminated = False

            for t in itertools.count():

                # Animate the last episode
                if episode == self.n_episode - 1:
                    self.env.render()

                A = self.policy(pi[S])
                obs, R, terminated, truncated = self.env.step(A)
                next_S = obs[0], obs[1], obs[2], obs[3], obs[4], obs[5]
                
                score += R
                sum_reward[episode] = score
                num_step[episode] = t

                delta = R + (self.gamma * self.V(next_S, w)) - self.V(S, w)

                if terminated:
                    break

                z_w = (self.gamma * (self.lamda**w) * z_w) + self.fourier_basis(S)
                z_theta = (self.gamma * (self.lamda**theta) * z_theta) + (I * self.dln(pi, S, A))
                w += (self.alpha**w) * delta * z_w

                theta += (self.alpha**theta) * delta * z_theta

                I = self.gamma * I
                S = next_S
                
        return pi, sum_reward, num_step