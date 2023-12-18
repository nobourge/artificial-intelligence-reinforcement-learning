from lle import Agent, ObservationType
from rlenv import RLEnv, Observation
import numpy as np
import numpy.typing as npt
import sys
from mdp import MDP, S, A
from lle import LLE, Action, Agent, AgentId, WorldState
from rlenv.wrappers import TimeLimit
from qagent import QAgent
from solution import Solution
from scores_displayer import ScoresDisplayer
from visualize import display_solution
from world_mdp import WorldMDP


class QLearning:
    """Tabular QLearning"""

    def __init__(
        self,
        lle,
        mdp: MDP[S, A],
        learning_rate: float,  # rate of learning
        discount_factor: float,  # favoring future rewards
        noise: float,  # exploration rate = noise
        seed: int = None,
        level: int = None,
    ):
        # Initialize parameters
        self.level = level
        self.mdp = mdp
        if lle:
            self.lle = lle
        elif level:
            self.lle = LLE.level(level, ObservationType.LAYERED)
        initial_observation = self.lle.reset()

        # Create the agents
        self.qagents = [
            QAgent(
                initial_observation,
                mdp,
                learning_rate,
                discount_factor,
                noise,
                id=AgentId(i),
            )
            for i in range(mdp.world.n_agents)
        ]

    def print_episode_results(
        self, episode, actions_taken=None, done=False, truncated=False, score=None
    ):
        """Print the results of the episode"""
        # Print the result of the episode
        if done:
            print(f"Episode {episode + 1} finished. ")
        elif truncated:
            print(f"Episode {episode + 1} truncated.")
        else:
            print(f"Episode {episode + 1} did not finish.")
        if actions_taken:
            print(f"Actions taken: {actions_taken}")
            print(f"Actions taken quantity: {len(actions_taken)}")
        print("score:", score)

    def train(
        self,
        agents,
        episodes_quantity: int,
        step_limit: int = 80,
    ):
        """Train the agent for the given number of episodes"""
        if self.level:
            env = TimeLimit(
                LLE.level(1, ObservationType.LAYERED),
                step_limit,
            )  # Maximum 80 time steps         # from instructions
        elif self.lle:
            env = TimeLimit(self.lle, 80)
        scores = []
        for episode in range(episodes_quantity):
            # print("episode:", episode)
            observation = env.reset()  # from instructions
            done = truncated = False  # from instructions
            actions_taken = []
            score = 0  # from instructions
            while not (done or truncated):  # from instructions
                actions = [  # from instructions
                    a.choose_action(
                        observation,
                        episodes_quantity=episodes_quantity,
                        current_episode=episode,
                    )
                    for a in agents  # from instructions
                ]  # from instructions
                actions_taken.append(actions)
                next_observation, reward, done, truncated, info = env.step(
                    actions
                )  # from instructions
              
                for a in agents:  # from instructions
                    a.update(  # from instructions
                        observation, actions[a.id], reward, next_observation
                    )
                score += reward  # from instructions
                observation = next_observation
            scores.append(score)
        return agents, scores

    def test(
        self, env: RLEnv, trained_agents, episodes_quantity: int, save: bool = False
    ):
        """Test the agent for the given number of episodes"""
        for episode in range(episodes_quantity):
            # Reset the environment
            observation = env.reset()
            done = False
            actions_taken = []
            step_number = 0
            while not done:
                step_number += 1
                actions = [
                    a.choose_action(observation, training=False) for a in trained_agents
                ]
                print("step ", step_number, " actions:", actions)
                actions_taken.append(actions)
                next_observation, reward, done, truncated, info = env.step(actions)
                observation = next_observation
            # Print the result of the episode
            if done:
                print(f"Episode {episode + 1} finished. Actions taken: {actions_taken}")
                if save:
                    # Save the actions taken
                    with open(f"actions_taken_{episode + 1}.txt", "w") as f:
                        f.write(str(actions_taken))
                return actions_taken
            else:
                print(f"Episode {episode + 1} did not finish.")

   