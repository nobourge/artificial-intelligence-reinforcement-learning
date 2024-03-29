from dataclasses import (
    dataclass,
)  
@dataclass
class Parameters:
    """Parameters for the MDP and the algorithm
    ModifiedRewardWorld.reward values"""
    reward_live: float
    """Reward for living at each time step"""
    gamma: float
    """Discount factor"""
    noise: float
    """Probability of taking a random action instead of the chosen one"""

def prefer_close_exit_following_the_cliff() -> Parameters:
    # A strategy focusing on reaching the closest exit quickly,
    # even if it means taking risks.
    return Parameters(
        reward_live=-1,  # negative to encourage speed,
        gamma=0.1,  # enough to value exit but not too much for preffering the close exit to the far one
        noise=0,  # low to avoid random actions and exploration
    )

def prefer_close_exit_avoiding_the_cliff() -> Parameters:
    # A cautious strategy aiming for the nearest exit while avoiding risks.
    return Parameters(
        reward_live=-1, 
        gamma=0.1, 
        noise=0.5  # noise is higher to encourage exploration
    )

def prefer_far_exit_following_the_cliff() -> Parameters:
    # A strategy that targets a distant exit but might involve risk-taking.
    return Parameters(
        reward_live=0,  # no hurry
        gamma=0.8,  # high for focusing on distant rewards,
        noise=0.2,  # moderate for some randomness in path selection
    )

def prefer_far_exit_avoiding_the_cliff() -> Parameters:
    # A strategy preferring a distant exit with an emphasis on safety and planning.
    # Less negative reward_live for longer routes,
    # high gamma for future-oriented planning, and
    # low noise for consistent decision-making.
    return Parameters(reward_live=1, 
                      gamma=0.8, 
                      noise=0.9
                      )

def never_end_the_game() -> Parameters:
    # A unique strategy to avoid reaching terminal states and keep the game ongoing.
    return Parameters(
        reward_live=11,  # reward_live bigger than biggest reward to encourage continual play
        gamma=0,
        noise=0,
    )
