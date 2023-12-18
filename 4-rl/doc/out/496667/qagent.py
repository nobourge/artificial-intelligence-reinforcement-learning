import numpy as np
from rlenv import Observation
from typing import Dict, Tuple, List, Iterable, Generic, Optional, Callable, Set
from mdp import MDP, S, A
from qvalues_displayer import QValuesDisplayer
from world_mdp import WorldMDP
from lle import LLE, Action, Agent, AgentId, Position, WorldState

class QAgent:
    def __init__(
        self,
        # env: RLEnv,
        # world_size: int,
        initial_observation: Observation,
        mdp: MDP[S, A],
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 0.1,
        seed: int = None,
        id: AgentId = None,
    ):
        """Initialize the agent"""
        # Initialize the environment
        self.analyse_observation(initial_observation, initial=True)
        # Initialize the MDP
        self.mdp = mdp
        # Initialize parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.id = id
        # Initialize Q-table as a dictionary
        # Pour favoriser l’exploration, initialisez vos 𝑄(𝑠, 𝑎) à 1 et non à 0
        self.q_table = {
            observation: {action.value: 1 for action in Action.ALL}
            for observation in mdp.states()
        }  # dict of dicts

        self.qvalues_displayer = QValuesDisplayer(self.world_size, self.q_table)

        # Initialize a random number generator
        self.rng = np.random.default_rng(seed)  # Random number generator instance

    def observe(self, observation: Observation):
        """Observe the given observation"""
        # Lorsque vous récupérez une observation, vous pouvez accéder à son contenu avec observation.data
        # qui contient un tableau numpy dont la forme est (n_agents, ...)
        observation_data = observation.data

    def get_ones_indexes(
        self,
        array: np.ndarray,
    ) -> List[int]:
        """Get the indexes of all ones in the given array"""
        ones_indexes = []
        for i, value in enumerate(array):
            if value == 1:
                ones_indexes.append(i)
        return ones_indexes

    def choose_action(
        self,
        observation: Observation,  # from instructions
        training: bool = True,
        episodes_quantity: int = 100,
        current_episode: int = 0,
    ):
        """Choose an action using the epsilon-greedy policy"""
        exploitation = False
        exploration = False
        if training:
            # Update epsilon
            epsilon = self.epsilon * (1 - current_episode / episodes_quantity)
            if epsilon < self.rng.uniform(0, 1):
                exploitation = True
            else:
                exploration = True
        if not training or exploitation:
            # Exploitation: Best known action
            observation_actions = self.q_table.get(observation, {})
            if observation_actions:
                action = max(observation_actions, key=observation_actions.get)
            else:
                exploration = True
        if exploration:
            observation_available_actions = observation.available_actions
            current_agent_available_actions = observation_available_actions[self.id]
            valid_actions = self.get_ones_indexes(current_agent_available_actions)
            action = self.rng.choice(valid_actions)

        return action

    def get_dangerosity(self, lasers) -> float:
        """Return the dangerosity of the given MDP"""
        dangerous_cells_quantity = 0
        # remove matrix with index of the agent id
        dangerous_lasers_matrices_indexes = lasers[: self.id] + lasers[self.id + 1 :]
        for matrix in dangerous_lasers_matrices_indexes:
            dangerous_cells_quantity += np.count_nonzero(matrix)
        dangerosity = dangerous_cells_quantity / len(self.not_wall_positions)
        return dangerosity

    def adapt_reward_live(
        self,
    ) -> float:
        """Return the reward_live of the given MDP
        if the world is dangerous, the reward_live is null
        if the world is safe, the reward_live is negative
        """
        self.reward_live = self.dangerosity - 1

    def adapt_learning_rate(
        self,
    ):
        """Adapt the learning rate to the given MDP"""
        # if the world is dangerous, increase the learning rate
        self.learning_rate = 0.2 + self.dangerosity

    def adapt_discount_factor(
        self,
    ):
        """Adapt the discount factor to the given MDP"""
        # if the world is dangerous, increase the discount factor
        self.discount_factor = 0.9 - self.dangerosity

    def adapt_learning_parameters(
        self,
        lasers: list,
    ):
        """Adapt the learning parameters to the given MDP"""
        # if the world is dangerous, increase the learning rate
        self.dangerosity = self.get_dangerosity(lasers)
        self.adapt_reward_live()
        self.adapt_learning_rate()
        self.adapt_discount_factor()

    def analyse_observation(
        self,
        observation: Observation,
        initial: bool = False,
    ):
        observation_data = observation.data
        observation_data_list = observation_data[0]
        # print("observation_data:\n", observation_data_list)
        observation_shape = observation_data.shape
        if initial:
            self.agents_quantity = observation_shape[0]
            self.world_size = observation_shape[1] * observation_shape[2]
            self.exits = np.transpose(np.nonzero(observation_data_list[-1]))
            self.walls = np.transpose(
                np.nonzero(observation_data_list[self.agents_quantity])
            )
            self.not_wall_positions = np.argwhere(observation_data_list[self.agents_quantity] == 0)
            self.not_wall_positions_quantity = len(self.not_wall_positions)
        lasers_matrices_list = [
            np.transpose(np.nonzero(layer))
            for layer in observation_data_list[self.agents_quantity:-2]
        ]
        self.gems = np.transpose(np.nonzero(observation_data_list[-2]))
        return lasers_matrices_list

    def update(self, observation, action, reward, next_observation):
        """Update the Q-table using the Bellman equation adapted for Q-learning:
        𝑄(𝑠, 𝑎) ← (1 − 𝛼)𝑄(𝑠, 𝑎) + 𝛼[𝑅(𝑠, 𝑎, + 𝛾𝑉 (𝑠′)] """
        lasers_matrices_list = self.analyse_observation(observation)
        self.adapt_learning_parameters(
            lasers_matrices_list,
        )
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
