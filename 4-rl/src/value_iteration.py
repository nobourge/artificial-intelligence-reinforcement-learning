import copy
import sys

from lle import World, WorldState
from almost_equal import almost_equal
from graph_mdp import GraphMDP
from mdp import MDP, S, A
from typing import Generic

from auto_indent import AutoIndent
from world_mdp import WorldMDP
import utils

sys.stdout = AutoIndent(sys.stdout)


class ValueIteration(Generic[S, A]):
    def __init__(self, mdp: MDP[S, A], gamma: float):  # discount factor
        # senf.values est nécessaire pour fonctionner avec utils.show_values
        self.mdp = mdp
        self.gamma = gamma
        # self.values = dict[S, float]()
        self.values = {
            state: 0.0 for state in mdp.states()
        }  # Initialize all states with a default value
        # utils.show_values(self.values)
        # utils.

    def value(self, state: S) -> float:
        """Returns the value of the given state."""
        # return self.values[state]
        # return self.values.get(state, 0.0)  # Default value if state not found
        if state not in self.values:
            return 0.0
        return self.values.get(state)

    def policy(self, state: S) -> A:
        """Returns the action
        that maximizes the Q-value of the given state."""
        available_actions = self.mdp.available_actions(state)
        if not available_actions:
            print("No available actions for state", state)
            return None  # Or some default action if appropriate
        return max(available_actions, key=lambda action: self.qvalue(state, action))

    def qvalue(self, state: S, action: A) -> float:
        """
        Returns the Q-value
        of the given state-action pair
        based on the state values.
        from Bellman equation:
        Q(s,a) = Sum(P(s,a,s') * (R(s,a,s') + gamma * V(s')))
        """
        qvalue = 0.0
        next_states_and_probs = self.mdp.transitions(state, action)
        # print("next_states_and_probs: \n", next_states_and_probs, "\n")
        for next_state, prob in next_states_and_probs:
            reward = self.mdp.reward(state, action, next_state)
            # print("P(", state, action, next_state, "):", prob)
            # print("R(", state, action, next_state, "):", reward)
            next_state_value = self.value(next_state)
            # print("V(", next_state, "):", next_state_value)
            qvalue += prob * (reward + self.gamma * next_state_value)
        # print("Q-value of", state, action, ":", qvalue, "\n")
        return qvalue

    def _compute_value_from_qvalues(self, state: S) -> float:
        """
        Returns the value of the given state based on the Q-values.
        from Bellman equation:
        V(s) = max_a Sum(P(s,a,s') * (R(s,a,s') + gamma * V(s')))

        This is a private method,
        meant to be used by the value_iteration method.
        """
        # print("Computing value of", state)
        value = max(
            self.qvalue(state, action) for action in self.mdp.available_actions(state)
        )
        if value is None:
            return 0.0
        return value

    def get_values_at_position(self, i: int, j: int) -> list[float]:
        """Returns the values of the states at the given position."""
        # world_gems_quantity = self.mdp.
        states_at_position = [
            state for state in self.mdp.states() if state.agents_positions[0] == (i, j)
        ]
        values_at_position = [self.value(state) for state in states_at_position]
        increasing_values = sorted(values_at_position)

        return increasing_values

    def print_values_table(self, n: int = 0):
        """In a map's representation table,
        each tile contains the possible values at that position."""
        if not isinstance(self.mdp, WorldMDP):
            print("Cannot print values table for non-world MDP")
            return None
        print("Iteration", n, "Values table: ")
        max_len = 0
        for i in range(self.mdp.world.height):
            for j in range(self.mdp.world.width):
                values = self.get_values_at_position(i, j)
                # Convert the list of values to a string and find the maximum length
                values_str = str(values)
                max_len = max(max_len, len(values_str))

        for i in range(self.mdp.world.height):
            for j in range(self.mdp.world.width):
                values = self.get_values_at_position(i, j)
                # Format each string to have the same width
                print(f"{str(values):<{max_len}}", end=" ")
            print()

    def print_iteration_values(self, iteration: int):
        """Prints the states and their values."""
        print("Iteration", iteration, "States and their values:")
        for state in self.mdp.states():
            print(state, self.value(state))

    def value_iteration(self, n: int):  # number of iterations
        """Performs value iteration for the given number of iterations."""
        for _ in range(n):
            # print("Iteration", _)
            new_values = copy.deepcopy(self.values)
            for state in self.mdp.states():  # All states generator (not a list)
                # print("State", state)
                if self.mdp.is_final(state):
                    # print("Final state", state)
                    new_values[state] = 0.0
                else:
                    new_values[state] = self._compute_value_from_qvalues(state)
            self.values = new_values
            self.print_values_table(_)
        self.print_iteration_values(n)


if __name__ == "__main__":
    # graph
    # b - +1 - a - -1 - c
    # graph_file_name = "tests/graphs/graph1.json"
    # mdp = GraphMDP.from_json(graph_file_name)
    # gamma = 0.9
    # algo = ValueIteration(mdp, gamma)
    # # algo.value_iteration(10)
    # algo.value_iteration(100)
    # assert almost_equal(algo.qvalue("a", "left"), 0.6)  # no change from iteration 0
    # assert almost_equal(
    #     algo.qvalue("a", "right"), 0.90909090909
    # )  # more than iteration 0 & 1

    mdp = WorldMDP(
        World(
            """
    .  . . X
    .  @ . V
    S0 . . ."""
        )
    )
    algo = ValueIteration(mdp, 0.9)
    # algo.value_iteration(10)
    algo.value_iteration(100)
    expected = [
        [1.62, 1.80, 2.0, 0.0],
        [1.458, 0.0, 1.80, 0.0],
        [1.3122, 1.458, 1.62, 1.458],
    ]
    for i in range(mdp.world.height):
        for j in range(mdp.world.width):
            print("State", (i, j))
            state = WorldState([(i, j)], [])
            print("Value:", algo.value(state))
            assert almost_equal(algo.value(state), expected[i][j])
