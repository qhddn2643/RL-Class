import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict, OrderedDict
from mpl_toolkits.axes_grid1 import make_axes_locatable

class n_step_SARSA:
    def __init__(self, env, n_episode, alpha, gamma, epsilon, n, tag) -> None:
        
        self.env = env
        self.n_episode = n_episode
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n = n
        self.tag = tag
        
    def epsilon_greedy_policy(self, Q, S):
        if np.random.rand() < self.epsilon:
            A = self.env.action_space.sample()
        else:
            A = np.argmax(Q[S])
        return A
        
    def play(self):

        Q = defaultdict(np.random.rand)
        sum_rewards = np.zeros(self.n_episode)
        policy = defaultdict(float)
        num_step = np.zeros(self.n_episode)
        
        for episode in range(self.n_episode):
            
            states = {}
            actions = {} 
            rewards = {}
            
            S = self.env.reset()            
            states[0] = S
            
            A = self.epsilon_greedy_policy(Q, states[0])
            actions[0] = A
            
            T = 1000
            t = 0
            tau = 0

            while True:

                if t < T:
                    next_S, R, terminated, truncated = self.env.step(actions[t % (self.n+1)])

                    states[(t+1) % (self.n+1)] = next_S
                    rewards[(t+1) % (self.n+1)] = R
                    sum_rewards[episode] += rewards[(t+1) % (self.n+1)]
                    num_step[episode] = t
                    
                    if terminated:
                        T = t + 1
                    else:
                        next_A = self.epsilon_greedy_policy(Q, states[(t+1) % (self.n+1)])
                        actions[(t+1) % (self.n+1)] = next_A
                        
                tau = t - self.n + 1

                if tau >= 0:
                    G = 0
                    for i in range(tau+1, min(tau + self.n, T)+1):
                        G += (self.gamma**(i-tau-1)) * rewards[i % (self.n+1)]

                    if tau + self.n < T:
                        G += np.power(self.gamma, self.n)*Q[states[(tau+self.n) % (self.n+1)], actions[(tau+self.n) % (self.n+1)]]

                    Q[states[tau % (self.n+1)], actions[tau % (self.n+1)]] += self.alpha * (G - Q[states[tau % (self.n+1)], actions[tau % (self.n+1)]])
                    
                    policy[states[tau % (self.n+1)]] = self.epsilon_greedy_policy(Q, states[tau % (self.n+1)])
                
                t += 1
                
                if tau == T - 1:
                    break
        
        return Q, policy, sum_rewards, num_step
    
    
    
    def plot_value_function(self, Q):
        
        if self.tag == "Blackjack":
            
            fig, axes = plt.subplots(nrows=2, figsize=(25, 40), subplot_kw={'projection': '3d'})
            axes[0].set_title('Value Function without usable ace')
            axes[1].set_title('Value Function with usable ace')

            player_sum = np.arange(12, 22)
            dealer_show = np.arange(1, 11)
            usable_ace = np.array([False, True])
            Z = np.zeros((len(player_sum), len(dealer_show), len(usable_ace)))

            for i, player in enumerate(player_sum):
                for j, dealer in enumerate(dealer_show):
                    for k, ace in enumerate(usable_ace):
                        Z[i, j, k] = Q[player, dealer, ace]

            X, Y = np.meshgrid(dealer_show, player_sum)

            axes[0].plot_wireframe(X, Y, Z[:, :, 0])
            axes[1].plot_wireframe(X, Y, Z[:, :, 1])

            for ax in axes[0], axes[1]:
                ax.set_zlim(-1, 1)
                ax.set_xlabel('Dealer Showing')
                ax.set_ylabel('Player Sum')
                ax.set_zlabel('State_Action')
                
        else:
            
            fig, axes = plt.subplots(nrows=1, figsize=(25, 40), subplot_kw={'projection': '3d'})
            axes.set_title('Value Function of %s' % self.tag)

            states = range(self.env.observation_space.n)
            actions = range(self.env.action_space.n)
            Z = np.zeros((len(states), len(actions)))

            for i, s in enumerate(states):
                for j, a in enumerate(actions):
                    Z[i, j] = Q[s, a]

            X, Y = np.meshgrid(actions, states)

            axes.plot_wireframe(X, Y, Z)

            axes.set_xlabel('Action')
            axes.set_ylabel('State')
            axes.set_zlabel('State_Action Value')
        
        
            
    def print_optimal_policy(self, Q):
        
        states = []
        new_actions = []
        
        if self.tag == "Blackjack":
            def get_Z(player_hand, dealer_showing, usable_ace):
                if (player_hand, dealer_showing, usable_ace) in Q:
                    return Q[player_hand, dealer_showing, usable_ace] 
                else:
                    return 1

            def get_figure(usable_ace, ax):
                x_range = np.arange(1, 11)
                y_range = np.arange(11, 22)
                X, Y = np.meshgrid(x_range, y_range)
                Z = np.array([[get_Z(player_hand, dealer_showing, usable_ace) for dealer_showing in x_range] for player_hand in range(21, 10, -1)])
                surf = ax.imshow(Z, cmap=plt.get_cmap('Accent', 2), vmin=0, vmax=1, extent=[0.5, 10.5, 10.5, 21.5])
                plt.xticks(x_range, ('A', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
                plt.yticks(y_range)
                ax.set_xlabel('Dealer Showing')
                ax.set_ylabel('Player Sum')
                ax.grid(color='black', linestyle='-', linewidth=1)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                cbar = plt.colorbar(surf, ticks=[0, 1], cax=cax)
                cbar.ax.set_yticklabels(['0 (STICK)','1 (HIT)'])
                cbar.ax.invert_yaxis() 
                
            fig = plt.figure(figsize=(15, 15))
            ax = fig.add_subplot(121)
            ax.set_title('Usable Ace', fontsize=16)
            get_figure(True, ax)
            ax = fig.add_subplot(122)
            ax.set_title('No Usable Ace', fontsize=16)
            get_figure(False, ax)
            plt.show()
            
        else:
            print("Optimal Policy of", self.tag)
                        
            if self.tag == "Taxi":
                actions = [u'\u2193', u'\u2191', u'\u2192', u'\u2190', "P", "D"] # South, North, East, West, Pick-up, Drop-off    
                rows = 5
                cols = 5
                size = (rows, cols)
            if self.tag == "CliffWalking":
                actions = [ u'\u2191', u'\u2192', u'\u2190', u'\u2193'] # Up, Right, Down, Left
                rows = 4
                cols = 12
                size = (rows, cols)
            if self.tag == "FrozenLake":
                actions = [ u'\u2190', u'\u2193', u'\u2192', u'\u2191'] # Left, Down, Right, Up

                size = np.array(list(self.env.env.desc)).shape
                rows, cols = np.array(list(self.env.env.desc)).shape
                
            policy = np.array([np.argmax(Q[key]) if key in Q else -1 for key in np.arange(rows*cols)])
            print(np.take(actions, np.reshape(policy, size)))          

        
    def plot_rewards(self, rewards):
        plt.subplots()
        plt.plot(rewards, label="n=%d" % self.n)
        plt.xlabel("Episodes")
        plt.ylabel("Sum of Rewards")
        plt.title("Sum of Rewards during epsiode of %s" % self.tag)
        plt.legend(fontsize=7)
        plt.show()
        
        
    def plot_steps_episodes(self, num_step):
        plt.subplots()
        plt.plot(num_step, label="n=%d" % self.n)
        plt.xlabel("Episodes")
        plt.ylabel("Number of Step")
        plt.title("Number of Steps per episode of %s" % self.tag)
        plt.legend(fontsize=7)
        plt.show()