import numpy as np
import domain_mcst as dm

SIZE = int(input("Select a grid size: 5, 7, 10, 12, 15\nChoice: "))
dm.SIZE = SIZE

state = dm.Node()

print("Starting Grid: \n", dm.state_string(state.grid.astype(str)))
print(state.grid)
dm.valid_actions(state)
# gauge sub-optimality with rollouts
num_rollouts = 10000
# node = dm.Node(grid)
# for r in range(num_rollouts):
#    dm.rollout(node)
#    if r % (num_rollouts // 10) == 0: print(r, node.score_estimate)
