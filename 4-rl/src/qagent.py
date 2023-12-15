# qlearning inheriting agent

from itertools import product
import numpy as np
import random
import sys
import os
import time
from rlenv import Observation, RLEnv
from typing import Dict, Tuple, List, Iterable, Generic, Optional, Callable, Set
from mdp import MDP, S, A
from auto_indent import AutoIndent
from world_mdp import WorldMDP
import utils
from lle import LLE, Action, Agent, AgentId, Position, WorldState
from rlenv.wrappers import TimeLimit

# import Action class :


sys.stdout = AutoIndent(sys.stdout)


# class QAgent(QLearning):
class QAgent:
    def __init__(
        self,
        # env: RLEnv,
        mdp: MDP[S, A],
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 0.1,
        seed: int = None,
        id: AgentId = None,
    ):
        # Initialize the MDP
        self.mdp = mdp
        # Initialize parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.id = id
        print("self.id:", self.id)

        # Initialize the environment
        # self.env = env

        # Initialize Q-table as a dictionary
        # Pour favoriser l’exploration, initialisez vos 𝑄(𝑠, 𝑎) à 1 et non à 0
        self.q_table = {
            observation: {action.value: 1 for action in Action.ALL}
            # for observation in mdp.observations()
            for observation in mdp.states()
        }  # dict of dicts

        # Initialize a random number generator
        self.rng = np.random.default_rng(seed)  # Random number generator instance

    def observe(self, observation: Observation):
        """Observe the given observation"""
        # Lorsque vous récupérez une observation, vous pouvez accéder à son contenu avec observation.data
        # qui contient un tableau numpy dont la forme est (n_agents, ...)

        observation_data = observation.data
        print("observation_data:", observation_data)

    def get_position(self, matrix: np.ndarray) -> np.ndarray:
        """Get the position of the agent"""
        # return np.where(matrix == 1)

        print("matrix:", matrix)
        print("np.nonzero(matrix):", np.nonzero(matrix))
        print("np.transpose(np.nonzero(matrix)):", np.transpose(np.nonzero(matrix)))
        return np.transpose(np.nonzero(matrix))

    def get_ones_indexes(
        self,
        array: np.ndarray,
    ) -> List[int]:
        """Get the indexes of all ones in the given array"""
        ones_indexes = []
        for i, value in enumerate(array):
            if value == 1:
                ones_indexes.append(i)

        # print("ones_indexes:", ones_indexes)
        return ones_indexes

    def choose_action(
        self,
        observation: Observation,  # from instructions
    ):
        """Choose an action using the epsilon-greedy policy"""
        observation_available_actions = observation.available_actions
        # print("observation_available_actions:", observation_available_actions)
        current_agent_available_actions = observation_available_actions[self.id]

        valid_actions = self.get_ones_indexes(current_agent_available_actions)
        # print("valid_actions:", valid_actions)
        if self.rng.uniform(0, 1) < self.epsilon:
            # Exploration: Random Action

            action = self.rng.choice(valid_actions)
        else:
            # Exploitation: Best known action
            observation_actions = self.q_table.get(observation, {})
            if observation_actions:
                action = max(observation_actions, key=observation_actions.get)
            else:
                action = self.rng.choice(valid_actions)
        return action
        # return Action(action)

    def update(self, observation, action, reward, next_observation):
        """Update the Q-table using the Bellman equation adapted for Q-learning:
        𝑄(𝑠, 𝑎) ← (1 − 𝛼)𝑄(𝑠, 𝑎) + 𝛼[𝑅(𝑠, 𝑎, 𝑠′ + 𝛾𝑉 (𝑠′)] """
        # Get the current Q value
        current_q = self.q_table.get(observation, {}).get(
            action, 1
        )  # 1 = default value
        # Find the max Q value for the actions in the next observation
        next_observation_actions = self.q_table.get(next_observation, {})
        max_next_q = max(next_observation_actions.values(), default=0)
        # Update the Q value using the Bellman equation
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q
        )
        # Update the Q-table
        self.q_table.setdefault(observation, {})[action] = new_q

    def print_q_table(self):
        """Print the Q-table as a table"""
        pass


if __name__ == "__main__":
    # Create the environment
    env = LLE.level(1)
    # Create the MDP
    mdp = WorldMDP(env.world)
    print(mdp.world)

    # Create the agents
    agent = QAgent(mdp, AgentId(1))

    # # Train the agent
    # agent.train(env, episodes_quantity=100)
    # # Test the agent
    # agent.test(env, episodes_quantity=100)
    # # Save the agent
    # agent.save(
    #     "qlearning_agent.pkl"
    # )  # pkl = pickle = sérialisation de données en Python
