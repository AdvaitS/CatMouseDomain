import numpy as np
import domain_mcst as dm

EMPTY, CAT, MOUSE, WALL, TRAP, HOLE = list(range(6))
SIZE = 7


def putagents(grid):
    grid_copy = grid.flatten()
    pos = [i for i in range(len(grid_copy)) if grid_copy[i] == EMPTY]
    m, c, t, h = np.random.choice(pos, size=4)
    grid_copy[m], grid_copy[c], grid_copy[t], grid_copy[h] = MOUSE, CAT, TRAP, HOLE
    return grid_copy.reshape((SIZE, SIZE))


grid = np.array([[EMPTY] * SIZE] * SIZE)
grid[SIZE // 2 - 1:SIZE // 2 + 2, SIZE // 2] = WALL
grid[SIZE // 2, SIZE // 2 - 1:SIZE // 2 + 2] = WALL
grid = putagents(grid)
print(grid)

# state[0,0] = "X" # optimal, according to https://xkcd.com/832/
# state[0,1] = "X" # suboptimal

# gauge sub-optimality with rollouts
# num_rollouts = 10000 #
# node = dm.Node(state)
# for r in range(num_rollouts):
#    dm.rollout(node)
#    if r % (num_rollouts // 10) == 0: print(r, node.score_estimate)
