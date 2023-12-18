
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class QValuesDisplayer:
    def __init__(self, 
                 world_size : int,
                 qtable : np.ndarray,
                 ):
        self.qtable = qtable
        self.board_size = world_size
        print("self.board_size:", self.board_size)

    def draw_board_with_triangles_and_values(self,
                                             ax, 
                                             values
                                             ):
        # Set the limits of the axis
        ax.set_xlim([0, self.board_size])
        ax.set_ylim([0, self.board_size])

        # Draw the board
        for x in range(self.board_size):
            for y in range(self.board_size):
                # Define the corners of the square
                square_corners = [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)]
                square_center = (x + 0.5, y + 0.5)

                # Draw 4 triangles within each square
                for i in range(4):
                    # Get the value for the current triangle
                    value = values[x][y][i]
                    triangle_corners = [
                        square_corners[i],
                        square_corners[(i + 1) % 4],
                        square_center,
                    ]
                    polygon = patches.Polygon(
                        triangle_corners,
                        closed=True,
                        edgecolor="black",
                        facecolor=value,
                    )
                    ax.add_patch(polygon)

                    # Calculate the center of the triangle for text placement
                    triangle_center = np.mean(triangle_corners, axis=0)
                    
                    # Place the value in the center of the triangle
                    ax.text(
                        *triangle_center, str(value), ha="center", va="center", fontsize=8
                    )

    def display_qvalues_board(self,
                              qvalues : np.ndarray
                              ):
        # Create a figure and axis
        _, ax = plt.subplots()

        # Define the board size and values for each triangle

        # Draw the board with values
        self.draw_board_with_triangles_and_values(ax, 
                                                  qvalues
                                                  )

        # Remove axis labels
        ax.axis("off")

        plt.show()
