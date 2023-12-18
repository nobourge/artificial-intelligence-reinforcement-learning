from src.value_iteration import ValueIteration
from lle import World, WorldState, Action, REWARD_AGENT_JUST_ARRIVED, REWARD_END_GAME
from tests.world_mdp import WorldMDP
from .graph_mdp import GraphMDP
from matplotlib import pyplot as plt


def almost_equal(a, b):
    return abs(a - b) < 1e-6


graph_file_name = "tests/graphs/graph1.json"

# 4-rl\tests\graphs\graph1.json
def test_value_0():
    mdp = GraphMDP.from_json(graph_file_name)
    algo = ValueIteration(mdp, 0.9)
    algo.value_iteration(0)
    for s in mdp.states():
        assert algo.value(s) == 0.0


def test_value_end_states():
    """
    tests that the value of end states is 0 after 100 iterations
    """
    mdp = GraphMDP.from_json(graph_file_name)
    algo = ValueIteration(mdp, 0.9)
    algo.value_iteration(100)
    for s in mdp.states():
        if mdp.is_final(s):
            assert algo.value(s) == 0.0


def test_qvalues_0():
    """
    test qvalues for the graph mdp after 0 iterations
    """
    mdp = GraphMDP.from_json(graph_file_name)
    algo = ValueIteration(mdp, 0.9)
    algo.value_iteration(0)
    assert almost_equal(algo.qvalue("a", "left"), 0.6)
    assert algo.qvalue("a", "right") == 0.5


def test_max_action_0():
    """
    test policy max_action for the graph mdp after 0 iterations
    """
    mdp = GraphMDP.from_json(graph_file_name)
    algo = ValueIteration(mdp, 0.9)
    algo.value_iteration(0)
    assert algo.policy("a") == "left"


def test_value_1():
    """
    test value of states after 1 iteration
    """
    mdp = GraphMDP.from_json(graph_file_name)
    gamma = 0.9
    algo = ValueIteration(mdp, gamma)
    algo.value_iteration(1)
    assert almost_equal(algo.qvalue("a", "left"), 0.6) # no change from iteration 0
    expected = 0.5 + 0.5 * gamma * 0.6 # 0.77 # = 0.5 + 0.5 * 0.9 * 0.6
    assert almost_equal(algo.qvalue("a", "right"), expected) # more than iteration 0


def test_value_100():
    """
    test value of states after 100 iterations
    """
    mdp = GraphMDP.from_json(graph_file_name)
    gamma = 0.9
    algo = ValueIteration(mdp, gamma)
    algo.value_iteration(100)
    assert almost_equal(algo.qvalue("a", "left"), 0.6) # no change from iteration 0
    assert almost_equal(algo.qvalue("a", "right"), 0.90909090909) # more than iteration 0 & 1


def test_value_world_0():
    """
    test value of world states after 0 iterations
    """
    mdp = WorldMDP(
        World(
            """
    .  . . X
    .  @ . V
    S0 . . ."""
        )
    )
    algo = ValueIteration(mdp, 0.9)
    algo.value_iteration(0)
    for s in mdp.states():
        assert algo.value(s) == 0.0


def test_qvalues_world():
    """
    test qvalues for the world mdp after 0 iterations
    """
    mdp = WorldMDP(
        World(
            """
    .  . . X
    .  @ . V
    S0 . . ."""
        )
    )
    algo = ValueIteration(mdp, 0.9)
    algo.value_iteration(0)
    state = WorldState([(0, 2)], [])
    assert (
        algo.qvalue(state, [Action.EAST]) == REWARD_END_GAME + REWARD_AGENT_JUST_ARRIVED
    )


def test_value_world_100():
    """
    test value of world states after 100 iterations
    """
    mdp = WorldMDP(
        World(
            """
    .  . . X
    .  @ . V
    S0 . . ."""
        )
    )
    algo = ValueIteration(mdp, 0.9)
    algo.value_iteration(100)
    expected = [
        [1.62, 1.80, 2.0, 0.0],
        [1.458, 0.0, 1.80, 0.0],
        [1.3122, 1.458, 1.62, 1.458],
    ]
    for i in range(mdp.world.height):
        for j in range(mdp.world.width):
            state = WorldState([(i, j)], [])
            assert almost_equal(algo.value(state), expected[i][j])



if __name__ == "__main__":
    print("hello world")
    test_value_0()
    test_value_end_states()
    test_qvalues_0()
    test_max_action_0()
    test_value_1()
    test_value_100()
    test_value_world_0()
    test_qvalues_world()
    test_value_world_100()
    print("ok")
