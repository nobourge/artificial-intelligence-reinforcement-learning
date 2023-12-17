# class ScoresDisplayer displays a list of scores per episode graph in a window

from matplotlib import pyplot as plt


class ScoresDisplayer:
    def __init__(self, scores, title):
        self.scores = scores
        self.title = title

    def display(self):
        plt.plot(self.scores)
        plt.title(self.title)
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.show()