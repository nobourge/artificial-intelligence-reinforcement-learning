from mdp import MDP, S, A
from typing import Generic


class ValueIteration(Generic[S, A]):
    def __init__(self, 
                 mdp: MDP[S, A], 
                 gamma: float # discount factor 
                 ):
        # senf.values est nécessaire pour fonctionner avec utils.show_values
        self.values = dict[S, float]()
        self.mdp = mdp
        self.gamma = gamma


    def value(self, state: S) -> float:
        """Returns the value of the given state."""
        return self.values[state]

    def policy(self, state: S) -> A:
        """Returns the action 
        that maximizes the Q-value of the given state."""
        return max(self.mdp.available_actions(state), key=lambda action: self.qvalue(state, action))

    def qvalue(self, state: S, action: A) -> float:
        """Returns the Q-value 
        of the given state-action pair based on the state values."""
        # return sum(prob * (reward + self.gamma * self.value(next_state)) for next_state, prob in self.mdp.transitions(state, action))
        return sum(prob * (self.gamma * self.value(next_state)) for next_state, prob in self.mdp.transitions(state, action))

    def _compute_value_from_qvalues(self, state: S) -> float:
        """
        Returns the value of the given state based on the Q-values.

        This is a private method, meant to be used by the value_iteration method.
        """
        return max(self.qvalue(state, action) for action in self.mdp.available_actions(state))

    def value_iteration(self, n: int):
        for _ in range(n):
            new_values = dict[S, float]()
            for state in self.mdp.states():
                if self.mdp.is_final(state):
                    new_values[state] = 0
                else:
                    new_values[state] = max(self.qvalue(state, action) for action in self.mdp.available_actions(state))
            self.values = new_values

