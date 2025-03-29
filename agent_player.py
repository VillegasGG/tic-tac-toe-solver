'''
This class represents the agent.
This player can:
 - Choose actions based on estimations
 - Record all the states and actions it has taken
 - Update estimations (after the game is over)
 - Save and load the policy
'''

import numpy as np
from math import inf
import pickle
class AgentPlayer:
    def __init__(self, name, epsilon=0.3):
        self.name = name
        self.states = []    # All positions taken during each game
        self.epsilon = epsilon
        self.gamma = 0.9   
        self.lr = 0.2
        self.state_values = {}

    def choose_action(self, env):
        actions = env.get_actions()
        player = env._player

        if np.random.uniform(0, 1) <= self.epsilon:
            action = np.random.choice(actions)

        else:
            max_value = -inf
            for a in actions:
                next_env = env.clone()
                next_env.step(a)
                board = tuple(next_env.get_observation(player).flatten())

                if self.state_values.get(board) is None:
                    value = 0
                else:
                    # print('State found!')
                    # self.showBoard(board)
                    # print('Value:', self.state_values[board])
                    value = self.state_values[board]

                if value > max_value:
                    max_value = value
                    action = a

        return action
    
    def addState(self, state):
        self.states.append(state)
    
    def feedReward(self, reward):
        for state in reversed(self.states):
            if self.state_values.get(state) is None:
                self.state_values[state] = 0
            self.state_values[state] += self.lr * (self.gamma * reward - self.state_values[state])
            reward = self.state_values[state]

    def savePolicy(self, name):
        file_w = open(name, 'wb')
        pickle.dump(self.state_values, file_w)
        file_w.close()

    def loadPolicy(self, name=''):
        file_r = open(name, 'rb')
        self.state_values = pickle.load(file_r)
        file_r.close()
        print('Policy loaded')

    def showBoard(self, tuple):
        board = np.array(tuple).reshape(3, 3)
        print(board)

    def showPolicy(self):
        print('Policy: ')
        for key in self.state_values:
            self.showBoard(key)
            print('Value:', self.state_values[key])
            print('-------------------')





