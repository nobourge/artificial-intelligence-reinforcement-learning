# qlearning inheriting agent

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
from lle import LLE, Action, Agent, AgentId, WorldState
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
        _agent_id: AgentId = None,
    ):
        # Initialize the MDP
        self.mdp = mdp
        # Initialize parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self._agent_id = _agent_id

        # Initialize the environment
        # self.env = env

        # Initialize Q-table as a dictionary
        # Pour favoriser l’exploration, initialisez vos 𝑄(𝑠, 𝑎) à 1 et non à 0
        self.q_table = {
            self.hash(state, 
                      action): 1
            # self.hash(self.get_state_observation(state),
                    #    action): 1
            for state in mdp.states()
            # for observation in mdp.states()
            for action in Action.ALL
        }  # dict of dicts

        # Initialize a random number generator
        self.rng = np.random.default_rng(seed)  # Random number generator instance

    def hash(self, state: WorldState, action: Action) -> Tuple[WorldState, Action]:
        return (state, action.value)

    def observe(self, observation: Observation):
        """Observe the given observation"""
        # Lorsque vous récupérez une observation, vous pouvez accéder à son contenu avec observation.data
        # qui contient un tableau numpy dont la forme est (n_agents, ...)

        observation_data = observation.data
        print("observation_data:", observation_data)

    def get_state_observation(self,
                              state: WorldState,
                                ) -> Observation:
        """Get the observation for the given state"""
        return Observation(state)
    
    def choose_action(self, 
                      observation: Observation,
                      ):
        """Choose an action using the epsilon-greedy policy"""
        if self.rng.uniform(0, 1) < self.epsilon:
            # Exploration: Random Action
            action = self.rng.choice(Action.ALL)
        else:
            # Exploitation: Best known action
            state_actions = self.q_table.get(observation, {})
            if state_actions:
                action = max(state_actions, key=state_actions.get)
            else:
                action = self.rng.choice(Action.ALL)
        return action


if __name__ == "__main__":
    # Create the environment
    env = LLE.level(1)
    # Create the MDP
    mdp = WorldMDP(env.world)

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
