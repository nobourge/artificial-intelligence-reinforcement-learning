from lle import Agent
from rlenv import RLEnv, Observation
import numpy as np
import numpy.typing as npt
import sys
from mdp import MDP, S, A


class QLearning:
    """Tabular QLearning"""

    def __init__(
        self,
        mdp: MDP[S, A],
        learning_rate: float,
        discount_factor: float,
        epsilon: float,
        actions: list,
        seed: int = None,
    ):
        # Initialize parameters
        self.mdp = mdp
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.actions = actions  # List of possible actions
        # Initialize Q-table as a dictionary
        # Pour favoriser l’exploration, initialisez vos 𝑄(𝑠, 𝑎) à 1 et non à 0
        self.q_table = {
            state: {action: 1 for action in actions} for state in mdp.states()
        }

        # Initialize a random number generator
        self.rng = np.random.default_rng(seed)  # Random number generator instance

    def choose_action(self, state):
        """Choose an action using the epsilon-greedy policy"""
        if self.rng.uniform(0, 1) < self.epsilon:
            # Exploration: Random Action
            action = self.rng.choice(self.actions)
        else:
            # Exploitation: Best known action
            state_actions = self.q_table.get(state, {})
            if state_actions:
                action = max(state_actions, key=state_actions.get)
            else:
                action = self.rng.choice(self.actions)
        return action

    def update(self, state, action, reward, next_state):
        """Update the Q-table using the Bellman equation"""
        # Get the current Q value
        current_q = self.q_table.get(state, {}).get(action, 0)
        # Find the max Q value for the actions in the next state
        next_state_actions = self.q_table.get(next_state, {})
        max_next_q = max(next_state_actions.values(), default=0)
        # Update the Q value using the Bellman equation
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        # Update the Q-table
        self.q_table.setdefault(state, {})[action] = new_q

    def train(self, env: RLEnv, episodes_quantity: int):
        """Train the agent for the given number of episodes"""
        # for _ in range(episodes_quantity):
        #     # Reset the environment
        #     state = env.reset()
        #     done = False
        #     while not done:
        #         # Choose an action
        #         action = self.choose_action(state)
        #         # Take the action
        #         next_state, reward, done = env.step(action)
        #         # Update the Q-table
        #         self.update(state, action, reward, next_state)
        #         # Update the state
        #         state = next_state
        

    def test(self, env: RLEnv, episodes_quantity: int):
        """Test the agent for the given number of episodes"""
        for _ in range(episodes_quantity):
            # Reset the environment
            state = env.reset()
            done = False
            while not done:
                # Choose an action
                action = self.choose_action(state)
                # Take the action
                next_state, _, done = env.step(action)
                # Update the state
                state = next_state

    def show(self):
        """Show the Q-table"""
        print(self.q_table)

    def __str__(self):
        """Return the Q-table as a string"""
        return str(self.q_table)

    def __repr__(self):
        """Return the Q-table as a string"""
        return str(self.q_table)

    def numpy_table_hash(self):
        """Return the hash of the Q-table as a numpy array"""
        # return np.array(list(self.q_table.items()), dtype=object).hash

        # using  hash(array.tobytes())
        return hash(np.array(list(self.q_table.items()), dtype=object).tobytes())
    
    
