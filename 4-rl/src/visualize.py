import cv2
from lle import LLE, World
from rlenv import RLEnv
#   import Solution class:
from solution import Solution



DISPLAY = True
SAVE = False


def display_world(name: str, world: "World"):
    if DISPLAY:
        img = world.get_image()
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.waitKey(1)
    if SAVE:
        cv2.imwrite(f"visualisations/{name}.png", img)

def display_solution(name: str, 
                     env: RLEnv,
                     solution : Solution
                     ):
    env.reset()
    world = env.world
    print("world:", world)
    display_world(name, world)

    for actions in solution.actions:
        print("actions:", actions)
        env.step(actions)
        display_world(name, world)

def display_solution_from_file(file_path: str):
    file_name = file_path.split("/")[-1]
    print("file_name:", file_name)
    with open(file_path, "r") as file:
        solution = Solution(eval(file.read()))
        print("solution:", solution)
        name = file_name.split(".")[0]
        display_solution(file_path, 
                         World(
                                LLE.level(name)
                         ), 
                         solution
                         )