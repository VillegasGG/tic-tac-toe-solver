'''
This class represents the agent.
This player can:
 - Choose actions based on estimations
 - Record all the states and actions it has taken
 - Update estimations (after the game is over)
 - Save and load the policy
'''

import numpy as np

class AgentPlayer:
    def __init__(self, name, epsilon=1):
        self.name = name
        self.states = []    # All positions taken during each game
        self.epsilon = epsilon
        self.gamma = 0.9    
        self.state_values = {}

    def choose_action(self, env):
        actions = env.get_actions()

        if np.random.uniform(0, 1) <= self.epsilon:
            action = np.random.choice(actions)
            
            return action
