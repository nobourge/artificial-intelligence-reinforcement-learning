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
            state: {action.value: 1 for action in Action.ALL} for state in mdp.states()
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

    def get_state_observation(
        self,
        state: WorldState,
    ) -> Observation:
        """Get the observation for the given state"""
        return Observation(state)

    def get_position(self, matrix: np.ndarray) -> np.ndarray:
        """Get the position of the agent"""
        # return np.where(matrix == 1)

        print("matrix:", matrix)
        print("np.nonzero(matrix):", np.nonzero(matrix))
        print("np.transpose(np.nonzero(matrix)):", np.transpose(np.nonzero(matrix)))
        return np.transpose(np.nonzero(matrix))

    def get_valid_actions(
        self,
        observation: Observation,
    ) -> List[Action]:
        """Get the list of valid actions for the given observation"""
        observation_data = observation.data
        print("observation_data:", observation_data)
        agent_position = self.get_position(observation_data[0][0])

        return world.available_actions()

    def agent_position_after_action(
        self, agent_pos: Position, action: Action
    ) -> Position:
        """The position of an agent after applying the given action."""
        try:
            print("agent_pos", agent_pos)
            print("action", action)
            agent_pos_after_action = agent_pos + action.delta
            print("agent_pos_after_action", agent_pos_after_action)
        except ValueError:
            raise ValueError("Invalid action")
        return agent_pos_after_action

    def are_valid_joint_actions(
        self, state: WorldState, joint_actions: Tuple[Action, ...]
    ) -> bool:
        """Whether the given joint actions are valid.
        an action is valid if it is available for an agent
        and if it does not lead the agent to be on the same position as another agent"""
        # print("are_valid_joint_actions()")
        # print("state", state)
        # print("joint_actions", joint_actions)
        # print("state.agents_positions", state.agents_positions)
        # # calculate agent positions after applying the joint action
        agents_positions_after_joint_actions = []
        for i, agent_pos in enumerate(state.agents_positions):
            agent_pos_after_action = self.agent_position_after_action(
                agent_pos, joint_actions[i]
            )
            agents_positions_after_joint_actions.append(agent_pos_after_action)
        return self.no_duplicate_in(agents_positions_after_joint_actions)

    def get_valid_joint_actions(
        self, state: WorldState, available_actions: Tuple[Tuple[Action, ...], ...]
    ) -> Iterable[Tuple[Action, ...]]:
        """Yield all possible joint actions that can be taken from the given state.
        Hint: you can use `self.world.available_actions()` to get the available actions for each agent.
        """
        # print("available_actions", available_actions)
        # cartesian product of the agents' actions
        for joint_actions in product(*available_actions):
            # print("joint_actions", joint_actions)

            if self.are_valid_joint_actions(state, joint_actions):
                yield joint_actions

    def get_agents_positions(
        self,
        #  state: np.ndarray
        observation: Observation,
    ) -> List[Position]:
        """Get the positions of all agents in the given observation.state"""
        observation_data = observation.data
        print("observation_data:", observation_data)
        state = observation.state
        print("state:", state)

        agents_positions = []
        for agent in self.mdp.world.agents:
            agent_position = self.get_position(state[agent.id])
            agents_positions.append(agent_position)

        return agents_positions

    def get_ones_indexes(
        self,
        array: np.ndarray,
    ) -> List[int]:
        """Get the indexes of all ones in the given array"""
        ones_indexes = []
        for i, value in enumerate(array):
            if value == 1:
                ones_indexes.append(i)
     
        print("ones_indexes:", ones_indexes)
        return ones_indexes

    def choose_action(
        self,
        observation: Observation,  # from instructions
    ):
        """Choose an action using the epsilon-greedy policy"""
        state = observation.state
        # #state type:
        # print("type(state):", type(state))
        # print("state:", state)
        # world = self.mdp.world
        # print("world:", world)

        # # valid_actions = self.get_valid_joint_actions(
        # # world_available_actions = self.mdp.world.available_actions()
        # world_available_actions = self.mdp.available_actions(observation.state)
        # print("world_available_actions:", world_available_actions)
        # valid_actions = self.mdp.world.available_actions()[self.id]
        # print("valid_actions:", valid_actions)
        # self.get_position(observation.data[0][0])
        observation_available_actions = observation.available_actions
        print("observation_available_actions:", observation_available_actions)
        current_agent_available_actions = observation_available_actions[self.id]

        valid_actions = self.get_ones_indexes(current_agent_available_actions)
        print("valid_actions:", valid_actions)
        if self.rng.uniform(0, 1) < self.epsilon:
            # Exploration: Random Action

            action = self.rng.choice(valid_actions)
        else:
            # Exploitation: Best known action
            state_actions = self.q_table.get(observation, {})
            if state_actions:
                action = max(state_actions, key=state_actions.get)
            else:
                action = self.rng.choice(valid_actions)
        return action

    def update(self, state, action, reward, next_state):
        """Update the Q-table using the Bellman equation"""
        # Get the current Q value
        current_q = self.q_table.get(state, {}).get(action, 1)  # 1 = default value
        # Find the max Q value for the actions in the next state
        next_state_actions = self.q_table.get(next_state, {})
        max_next_q = max(next_state_actions.values(), default=0)
        # Update the Q value using the Bellman equation
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        # Update the Q-table
        self.q_table.setdefault(state, {})[action] = new_q


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
