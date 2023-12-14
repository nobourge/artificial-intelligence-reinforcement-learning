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
        mdp: MDP[S, A],
        learning_rate: float,  # rate of learning
        discount_factor: float,  # favoring future rewards
        epsilon: float,  # exploration rate = noise
        seed: int = None,
        level: int = 1,
    ):
        # Initialize parameters
        self.level = level
        self.mdp = mdp
        self.env = LLE.level(level, ObservationType.LAYERED)
        initial_state = mdp.world.initial_state
        initial_observation = self.env.reset()
        initial_observation_data = initial_observation.data
        agents_quantity = mdp.world.n_agents
        print("agents_quantity:", agents_quantity)
        agents_positions = [agent.position for agent in initial_state.agents]
        print("agents_positions:", agents_positions)
        walls = np.transpose(
            np.nonzero(initial_observation_data[len(agents_positions)])
        )
        print("walls:", walls)
        lasers = [
            np.transpose(np.nonzero(layer))
            for layer in initial_observation_data[len(agents_positions) : -2]
        ]
        print("lasers:", lasers)
        exits = np.transpose(np.nonzero(initial_observation_data[-1]))
        print("exits:", exits)
        gems = np.transpose(np.nonzero(initial_observation_data[-2]))
        print("gems:", gems)

        self.adapt_learning_parameters(
            mdp, agents_quantity, agents_positions, lasers, exits, gems
        )
        # Create the agents
        self.agents = [
            QAgent(mdp, learning_rate, discount_factor, epsilon, id=AgentId(i))
            # for i in range(env.world.n_agents)
            for i in range(mdp.world.n_agents)
        ]

    # # Initialize a random number generator
    # self.rng = np.random.default_rng(seed)  # Random number generator instance

    def get_dangerosity(self, mdp: MDP[S, A], lasers) -> float:
        """Return the dangerosity of the given MDP"""
        world = mdp.world
        observation = Observation(world)
        # count the number of dangerous cells:
        dangerous_cells = 0
        for matrix in lasers:
            dangerous_cells += np.count_nonzero(matrix)

    def get_reward_live(
        self,
        mdp: MDP[S, A],
        agents_quantity: int,
        agents_positions: list,
        lasers: list,
        exits: list,
        gems: list,
    ) -> float:
        """Return the reward_live of the given MDP"""
        pass

    def adapt_learning_parameters(
        self,
        mdp: MDP[S, A],
        agents_quantity: int,
        agents_positions: list,
        lasers: list,
        exits: list,
        gems: list,
    ):
        """Adapt the learning parameters to the given MDP"""
        # if the world is dangerous, increase the learning rate
        dangerosity = self.get_dangerosity(lasers)
        self.reward_live = self.get_reward_live(
            mdp, agents_quantity, agents_positions, lasers, exits, gems
        )

    def train(self, agents, episodes_quantity: int):
        """Train the agent for the given number of episodes"""
        # from instructions:
        env = TimeLimit(
            LLE.level(1, ObservationType.LAYERED), 80
        )  # Maximum 80 time steps         # from instructions

        observation = env.reset()  # from instructions
        observation_data = observation.data
        observation_hash = self.numpy_table_hash(observation_data)
        print("observation_data:", observation_data)
        print("observation_hash:", observation_hash)

        done = truncated = False  # from instructions
        score = 0  # from instructions
        while not (done or truncated):  # from instructions
            actions = [  # from instructions
                a.choose_action(observation) for a in agents  # from instructions
            ]  # from instructions
            print("actions:", actions)
            # get action[0] type:
            print("type(actions[0]):", type(actions[0]))
            # north = Action(0)
            # print("north:", north)
            # south = Action(1)
            # print("south:", south)

            next_observation, reward, done, truncated, info = env.step(
                actions
            )  # from instructions
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
                print("actions:", actions)
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
    lle = LLE.level(1, ObservationType.LAYERED)
    env_world = lle.world
    mdp = WorldMDP(env_world)
    print("mdp.world :", mdp.world)

    # Train the agent
    qlearning = QLearning(
        mdp,
        #   learning_rate = 0.9,
        #   learning_rate = 0.7,
        #   learning_rate = 0.5,
        #   learning_rate = 0.3,
        learning_rate=0.2,
        #   learning_rate = 0.1,
        #   learning_rate = 0.01,
        discount_factor=0.9,
        epsilon=0.1
        #   epsilon=0
    )
    trained_agents = qlearning.train(qlearning.agents, episodes_quantity=100)

    # terminal prompt to continue:
    # input("Press Enter to continue...")

    # Test the agents
    actions_taken = qlearning.test(env, trained_agents, episodes_quantity=1, save=True)

    # display the solution:
    solution = Solution(actions_taken)
    print("solution:", solution)

    # display the solution:
    display_solution(
        "solution",
        #  env_world,
        env,
        solution,
    )

    # Save the agent
