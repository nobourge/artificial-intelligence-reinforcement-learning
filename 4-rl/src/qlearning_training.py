# Entraînez votre algorithme sur les niveaux 1, 3 et 6 de LLE. Dans votre rapport, créez un graphique
# pour chacun de ces trois niveaux qui montre le score (c’est-à-dire la somme des rewards par épisode)
# au cours de l’entrainement. 


from rlenv import RLEnv


def train_qlearning_agent(env: RLEnv, episodes_quantity: int):
    """Train the agent for the given number of episodes"""
    # Create the agent
    agent = QLearningAgent(env)
    # Train the agent
    agent.train(env, episodes_quantity)
    # Test the agent
    agent.test(env, episodes_quantity)
    # Return the agent
    return agent

