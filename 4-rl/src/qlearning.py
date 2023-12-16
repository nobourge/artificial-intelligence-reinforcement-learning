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
        epsilon: float,  # exploration rate = noise
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
        initial_observation_data = initial_observation.data
        initial_observation_data_list = initial_observation_data[0]
        print("initial_observation_data:\n", initial_observation_data_list)
        initial_observation_shape = initial_observation_data.shape
        agents_quantity = initial_observation_shape[0]
        self.world_size = initial_observation_shape[1] * initial_observation_shape[2]
        print("agents_quantity:", agents_quantity)

        agents_positions = mdp.world.agents_positions
        print("agents_positions:", agents_positions)
        self.walls = np.transpose(
            np.nonzero(initial_observation_data_list[agents_quantity])
            # get the walls from the observation
        )
        print("walls:", self.walls)
        self.not_wall_positions = np.transpose(
            # np.where(initial_observation_data_list[agents_quantity] == 0)
            np.argwhere(initial_observation_data_list[agents_quantity] == 0)
        )
        print("not_wall_positions:", self.not_wall_positions)
        self.not_wall_positions_quantity = len(self.not_wall_positions)
        lasers_matrices_list = [
            np.transpose(np.nonzero(layer))
            for layer in initial_observation_data_list[agents_quantity:-2]
        ]
        print("lasers_matrices_list:", lasers_matrices_list)
        self.exits = np.transpose(np.nonzero(initial_observation_data_list[-1]))
        print("exits:", self.exits)
        self.gems = np.transpose(np.nonzero(initial_observation_data_list[-2]))
        print("gems:", self.gems)

        self.adapt_learning_parameters(
            lasers_matrices_list,
        )
        # Create the agents
        self.qagents = [
            QAgent(
                   self.world_size,
                mdp,
                   learning_rate,
                    discount_factor, 
                    epsilon, 
                    id=AgentId(i)
                    )
            # for i in range(env.world.n_agents)
            for i in range(mdp.world.n_agents)
        ]

    # # Initialize a random number generator
    # self.rng = np.random.default_rng(seed)  # Random number generator instance

    def get_dangerosity(self, lasers) -> float:
        """Return the dangerosity of the given MDP"""
        dangerous_cells = 0
        for matrix in lasers:
            dangerous_cells += np.count_nonzero(matrix)

        dangerosity = dangerous_cells / len(self.not_wall_positions)
        print("dangerosity:", dangerosity)
        return dangerosity

    def get_reward_live(
        self,
        # mdp: MDP[S, A],
        # agents_quantity: int,
        # agents_positions: list,
        lasers: list,
        # exits: list,
        # gems: list,
    ) -> float:
        """Return the reward_live of the given MDP"""
        dangerosity = self.get_dangerosity(lasers)
        reward_live = dangerosity - 1
        print("reward_live:", reward_live)
        return reward_live

    def adapt_learning_parameters(
        self,
        # agents_quantity: int,
        # agents_positions: list,
        lasers: list,
        # gems: list,
    ):
        """Adapt the learning parameters to the given MDP"""
        # if the world is dangerous, increase the learning rate
        # dangerosity = self.get_dangerosity(lasers)
        self.reward_live = self.get_reward_live(
            # agents_quantity,
            # agents_positions,
            lasers,
            # gems
        )

    def episode_results(self, 
                        episode, 
                        actions_taken=None, 
                        done=False,
                        truncated=False
                        ):
        """Print the results of the episode"""
            # Print the result of the episode
        if done:
            print(f"Episode {episode + 1} finished. Actions taken: {actions_taken}")
        elif truncated:
            print(f"Episode {episode + 1} truncated.")
        else:
            print(f"Episode {episode + 1} did not finish.")

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
        for episode in range(episodes_quantity):
            observation = env.reset()  # from instructions
            # observation_data = observation.data
            # print("observation_data:", observation_data)
            done = truncated = False  # from instructions
            actions_taken = []
            score = 0  # from instructions
            while not (done or truncated):  # from instructions
                actions = [  # from instructions
                    a.choose_action(observation) for a in agents  # from instructions
                ]  # from instructions
                print("actions:", actions)
                print("type(actions[0]):", type(actions[0]))
                actions_taken.append(actions)
                next_observation, reward, done, truncated, info = env.step(
                    actions
                )  # from instructions
                if reward == 0:
                    reward = self.reward_live
                print("observation:", next_observation)
                print("reward:", reward)
                print("done:", done)
                print("truncated:", truncated)
                print("info:", info)

                for a in agents:  # from instructions
                    print("a:", a)
                    print("a.id:", a.id)
                    a.update(  # from instructions
                        observation, actions[a.id], reward, next_observation
                    )
                score += reward  # from instructions
                print("score:", score)
                observation = next_observation
            self.episode_results(episode, 
                                 actions_taken, 
                                 done, 
                                 truncated
                                 )
        return agents

    def test(
        self, env: RLEnv, trained_agents, episodes_quantity: int, save: bool = False
    ):
        """Test the agent for the given number of episodes"""
        for episode in range(episodes_quantity):
            # Reset the environment
            observation = env.reset()
            done = False
            actions_taken = []
            while not done:
                actions = [a.choose_action(observation) for a in trained_agents]
                # print("actions:", actions)
                actions_taken.append(actions)
                next_observation, reward, done, truncated, info = env.step(actions)
                print("reward:", reward)
                # print("done:", done)
                print("truncated:", truncated)
                print("info:", info)
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
    worldstr = """
    . S0 . X"""

    lle = LLE.from_str(worldstr, ObservationType.LAYERED)
    # lle = LLE.level(1, ObservationType.LAYERED)
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
        #   learning_rate = 0.3,
        learning_rate=0.2,
        #   learning_rate = 0.1,
        #   learning_rate = 0.01,
        discount_factor=0.9,
        epsilon=0.5
        # epsilon=0.1
        #   epsilon=0
    )
    trained_agents = qlearning.train(qlearning.qagents, episodes_quantity=100)

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
