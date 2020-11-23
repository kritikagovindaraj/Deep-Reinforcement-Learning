import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = 0.9
        self.gamma = 1.0

    def select_action(self, state, eps):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if np.random.random() > eps:
            return np.argmax(self.Q[state])
        else:
            return np.random.choice(np.arange(self.nA))
    
        

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        Qsa_next = np.max(self.Q[next_state]) if next_state is not None else 0
        A = reward + (self.gamma * Qsa_next)
        B = self.Q[state][action]
        new_value = B + (self.alpha * (A - B))
        self.Q[state][action] = new_value