from lle import LLE, Action, Agent, AgentId, Position, WorldState
from world_mdp import WorldMDP
from value_iteration import ValueIteration

from tests import world_mdp


def test_level_1():
    """
    test value of world states after 100 iterations
    """
    world = LLE.level(1).world
    mdp = WorldMDP(
        world
        
    )
    algo = ValueIteration(mdp, 0.9)
    algo.value_iteration(100)
    algo.show_values()
    