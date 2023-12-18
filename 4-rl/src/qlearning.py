from lle import Agent, ObservationType
from rlenv import RLEnv, Observation
import numpy as np
import numpy.typing as npt
import sys
from mdp import MDP, S, A
from lle import LLE, Action, Agent, AgentId, WorldState
from rlenv.wrappers import TimeLimit
from qagent import QAgent
from auto_indent import AutoIndent
from solution import Solution
from scores_displayer import ScoresDisplayer
from visualize import display_solution
from world_mdp import WorldMDP

sys.stdout = AutoIndent(sys.stdout)


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
            # for i in range(env.world.n_agents)
            for i in range(mdp.world.n_agents)
        ]

    # # Initialize a random number generator
    # self.rng = np.random.default_rng(seed)  # Random number generator instance

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
            # observation_data = observation.data
            # print("observation_data:", observation_data)
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
                # print("actions:", actions)
                actions_taken.append(actions)
                next_observation, reward, done, truncated, info = env.step(
                    actions
                )  # from instructions
                # if reward == 0:
                #     reward = self.reward_live #todo
                # print("observation:", next_observation)
                # print("reward:", reward)
                # print("done:", done)
                # print("truncated:", truncated)
                # print("info:", info)

                for a in agents:  # from instructions
                    # print("a:", a)
                    # print("a.id:", a.id)
                    a.update(  # from instructions
                        observation, actions[a.id], reward, next_observation
                    )
                score += reward  # from instructions
                observation = next_observation
            scores.append(score)
            # self.print_episode_results(episode,
            #                      actions_taken,
            #                      done,
            #                      truncated,
            # score
            #                      )
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
                # print("reward:", reward)
                # print("done:", done)
                # print("truncated:", truncated)
                # print("info:", info)
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

    def show(self):
        """Show the Q-table"""
        print(self.q_table)

    def __str__(self):
        """Return the Q-table as a string"""
        return str(self.q_table)

    def __repr__(self):
        """Return the Q-table as a string"""
        return str(self.q_table)

    def numpy_table_hash(self, numpy_table: npt.ArrayLike) -> int:
        """Return the hash of the Q-table as a numpy array"""
        # return np.array(list(self.q_table.items()), dtype=object).hash

        # using  hash(array.tobytes())
        return hash(np.array(list(numpy_table), dtype=object).tobytes())


if __name__ == "__main__":
    # Create the environment
    # worldstr = """
    # . S0 . X"""
    worldstr = """
    . S0 . X
    . S1 . X
    . . L0N ."""

    # lle = LLE.from_str(worldstr, ObservationType.LAYERED)
    # lle = LLE.level(1, ObservationType.LAYERED)
    # lle = LLE.level(3, ObservationType.LAYERED)
    lle = LLE.level(6, ObservationType.LAYERED)
    env_world = lle.world
    mdp = WorldMDP(env_world)
    print("mdp.world :", mdp.world)

    # Train the agent
    qlearning = QLearning(
        lle,
        mdp,
        #   learning_rate = 0.9,
        #   learning_rate = 0.7,
        #   learning_rate = 0.5,
        #   learning_rate = 0.4,
        #   learning_rate = 0.3,
        # learning_rate=0.2, # lvl1 perfect
        learning_rate=0.1,
        #   learning_rate = 0.01,
        # discount_factor=0.9, # lvl1 perfect
        discount_factor=0.5,
        # noise=0.8,
        # noise=0.6,
        # noise=0.5, #lvl1 perfect
        noise=0.4,
        # noise=0.2,
        # noise=0.1,
        #   noise=0,
    )
    trained_agents, scores = qlearning.train(qlearning.qagents, episodes_quantity=1000000)
    # trained_agents, scores = qlearning.train(qlearning.qagents, episodes_quantity=100)

    scores_displayer = ScoresDisplayer(scores, "Scores")
    scores_displayer.display()

    # terminal prompt to continue:
    # input("Press Enter to continue...")

    # Test the agents
    actions_taken = qlearning.test(lle, trained_agents, episodes_quantity=1, save=True)

    # display the solution:
    solution = Solution(actions_taken)
    print("solution:", solution)

    # display the solution:
    display_solution(
        "solution",
        #  env_world,
        lle,
        solution,
    )

    # # Save the agent
