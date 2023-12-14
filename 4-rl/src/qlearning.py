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
from world_mdp import WorldMDP

sys.stdout = AutoIndent(sys.stdout)


class QLearning:
    """Tabular QLearning"""

    def __init__(
        self,
        mdp: MDP[S, A],
        learning_rate: float,
        discount_factor: float,
        epsilon: float,
        seed: int = None,
    ):
        # Initialize parameters
        self.mdp = mdp
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        # Create the agents
        self.agents = [
            QAgent(mdp, learning_rate, discount_factor, epsilon, id=AgentId(i))
            # for i in range(env.world.n_agents)
            for i in range(mdp.world.n_agents)
        ]

    # # Initialize a random number generator
    # self.rng = np.random.default_rng(seed)  # Random number generator instance

    def train(self, agents, episodes_quantity: int):
        """Train the agent for the given number of episodes"""
        # from instructions:
        env = TimeLimit(
            LLE.level(1,
                      ObservationType.LAYERED
                      ), 
            80
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

            # south = Action(1)
            # print("south:", south)
            # north = Action(0)
            # print("north:", north)

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

    def numpy_table_hash(self, numpy_table: npt.ArrayLike) -> int:
        """Return the hash of the Q-table as a numpy array"""
        # return np.array(list(self.q_table.items()), dtype=object).hash

        # using  hash(array.tobytes())
        return hash(np.array(list(numpy_table), dtype=object).tobytes())


if __name__ == "__main__":
    # Create the environment
    env = LLE.level(1, ObservationType.LAYERED)
    mdp = WorldMDP(env.world)
    print("mdp.world :", mdp.world)

    # Train the agent
    agent = QLearning(mdp, 0.1, 0.9, 0.1)
    agent.train(agent.agents, episodes_quantity=100)
    # Test the agent
    # agent.test(env, episodes_quantity=100)
    # Save the agent
