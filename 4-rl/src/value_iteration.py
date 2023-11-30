from .mdp import MDP, S, A
from typing import Generic


class ValueIteration(Generic[S, A]):
    def __init__(self, mdp: MDP[S, A], gamma: float):  # discount factor
        # senf.values est nécessaire pour fonctionner avec utils.show_values
        self.mdp = mdp
        self.gamma = gamma
        # self.values = dict[S, float]()
        self.values = {
            state: 0.0 for state in mdp.states()
        }  # Initialize all states with a default value

    def value(self, state: S) -> float:
        """Returns the value of the given state."""
        # return self.values[state]
        return self.values.get(state, 0.0)  # Default value if state not found

    def policy(self, state: S) -> A:
        """Returns the action
        that maximizes the Q-value of the given state."""
        available_actions = self.mdp.available_actions(state)
        if not available_actions:
            print("No available actions for state", state)
            return None  # Or some default action if appropriate
        return max(available_actions, key=lambda action: self.qvalue(state, action))

    def bellman(self, state: S) -> float:
        """Returns the Bellman equation
        of the given state based on the state values."""
        return max(
            self.qvalue(state, action) for action in self.mdp.available_actions(state)
        ) # todo? compare with _compute_value_from_qvalues
    
    def qvalue(self, state: S, action: A) -> float:
        """Returns the Q-value
        of the given state-action pair based on the state values."""
        new_state = max(
            self.mdp.transitions(state, action), 
            key=lambda transition: transition[1])[0]  # Most probable next state
        reward = self.mdp.reward(state, action, new_state)
        return sum(
            prob * (reward + self.gamma * self.value(next_state))
            for next_state, prob in self.mdp.transitions(state, action)
        )

    def _compute_value_from_qvalues(self, state: S) -> float:
        """
        Returns the value of the given state based on the Q-values.

        This is a private method, meant to be used by the value_iteration method.
        """
        return max(
            self.qvalue(state, action) 
            for action in self.mdp.available_actions(state)
        )

    def show_iteration_values(self, iteration: int, states: list[S]):
        """Prints the map with each tile actions values."""
        print("image placeholder")

    def print_iteration_values(
        self, 
        iteration: int):
        """Prints the states and their values."""
        print("Iteration", iteration, "States and their values:")
        for state in self.mdp.states():
            print(state, self.value(state))

    def value_iteration(self, n: int):  # number of iterations
        """Performs value iteration for the given number of iterations."""
        states = self.mdp.states()

        for _ in range(n):
            new_values = dict[S, float]()
            for state in states:
                if self.mdp.is_final(state):
                    new_values[state] = 0
                else:
                    new_values[state] = self._compute_value_from_qvalues(state)
            self.values = new_values
        self.print_iteration_values(n)
