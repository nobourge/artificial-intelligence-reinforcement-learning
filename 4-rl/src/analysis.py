from dataclasses import (
    dataclass,
)  # dataclass is a decorator that allows you to create a class with a minimal amount of code 
# (see https://docs.python.org/3/library/dataclasses.html)


@dataclass
class Parameters:
    reward_live: float
    """Reward for living at each time step"""

    gamma: float
    """Discount factor"""
    noise: float
    """Probability of taking a random action instead of the chosen one"""

def prefer_close_exit_following_the_cliff() -> Parameters:
    # A strategy focusing on reaching the closest exit quickly, even if it means taking risks.
    # A small negative reward_live to encourage speed, moderate gamma for balancing immediate and future rewards, and low noise for deliberate actions.
    return Parameters(reward_live=-0.01, gamma=0.5, noise=0.1)


def prefer_close_exit_avoiding_the_cliff() -> Parameters:
    # A cautious strategy aiming for the nearest exit while avoiding risks.
    # Less negative reward_live for a safer, longer route, higher gamma for long-term planning, and low noise to avoid risky random moves.
    return Parameters(reward_live=-0.005, gamma=0.7, noise=0.05)


def prefer_far_exit_following_the_cliff() -> Parameters:
    # A strategy that targets a distant exit but might involve risk-taking.
    # Small negative reward_live, high gamma for focusing on distant rewards, and moderate noise for some randomness in path selection.
    return Parameters(reward_live=-0.01, gamma=0.8, noise=0.2)


def prefer_far_exit_avoiding_the_cliff() -> Parameters:
    # A strategy preferring a distant exit with an emphasis on safety and planning.
    # Less negative reward_live for longer routes, high gamma for future-oriented planning, and low noise for consistent decision-making.
    return Parameters(reward_live=-0.005, gamma=0.9, noise=0.05)


def never_end_the_game() -> Parameters:
    # A unique strategy to avoid reaching terminal states and keep the game ongoing.
    # Zero or slightly positive reward_live to encourage continual play, low gamma to de-emphasize distant futures (like exits), and high noise for unpredictability.
    return Parameters(reward_live=0.01, gamma=0.3, noise=0.5)


if __name__ == "__main__":
    print("Testing analysis.py")
