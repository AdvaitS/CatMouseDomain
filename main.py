import numpy as np
import domain_mcst as dm

SIZE = int(input("Select a grid size: 5, 7, 10, 12, 15\nChoice: "))
dm.SIZE = SIZE

state = dm.Node(dm.make_grid(), dm.CAT)

# print("Starting Grid: \n", dm.state_string(state.grid.astype(str)))
# print(state.grid)
# dm.valid_actions(state)
# gauge sub-optimality with rollouts
num_rollouts = 10000
# node = dm.Node(grid)
for r in range(num_rollouts):
    dm.explored = set()
    dm.rollout(state)
    if r % (num_rollouts // 10) == 0: print(r, state.score_estimate)

dm.explored = set()


def play_interactive(state):
    print("Start State:")
    print(dm.state_string(state.grid.astype(str)))
    while not dm.is_leaf(state):
        if state.turn == dm.CAT:
            print("Cat's Move")
            state = dm.exploit(state)
        else:
            print("Mouse' Move")
            state = dm.exploit(state)
        print(dm.state_string(state.grid.astype(str)))
    print("GAME OVER")


play_interactive(state)
