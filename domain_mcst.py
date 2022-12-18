# MCTS Implementation
import numpy as np
import mcst_nn as models
import torch as tr

EMPTY, CAT, MOUSE, WALL, TRAP, HOLE = [0, 1, 2, 3, 4, 5]
SIZE = -1
explored = {}
rollouts_visited = {}
rollouts = []


def winner(state):
    if state.mouse_pos == state.cat_pos:
        return CAT
    elif state.mouse_pos == tuple(np.argwhere(state.grid[HOLE] == 1)[0]):
        return MOUSE
    return -1


def state_string(grid):
    state = np.empty(grid[EMPTY].shape, dtype="<U10")
    for i in range(SIZE):
        for j in range(SIZE):
            if grid[EMPTY][i, j] == 1: state[i, j] = "_"
            if grid[WALL][i, j] == 1: state[i, j] = "W"
            if grid[TRAP][i, j] != 0: state[i, j] = "T"
            if grid[HOLE][i, j] == 1: state[i, j] = "H"
            if grid[CAT][i, j] == 1: state[i, j] += "C"
            if grid[MOUSE][i, j] == 1: state[i, j] += "M"
    return "\n".join(["  ".join(row) for row in state])


def get_actions(state):
    next_states = state.children()
    actions = []
    if state.turn == CAT:
        px, py = state.cat_pos
    else:
        px, py = state.mouse_pos
    for child in next_states:
        if state.turn == CAT:
            ix, iy = child.cat_pos
        else:
            ix, iy = child.mouse_pos
        if px - ix == 1 and py - iy == 0:
            actions.append("Upward")
        elif px - ix == 1 and py - iy == 1:
            actions.append("Upward Left")
        elif px - ix == 1 and py - iy == -1:
            actions.append("Upward Right")
        elif px - ix == -1 and py - iy == 0:
            actions.append("Downward")
        elif px - ix == -1 and py - iy == 1:
            actions.append("Downward Left")
        elif px - ix == -1 and py - iy == -1:
            actions.append("Downward Right")
        elif px - ix == 0 and py - iy == 0:
            actions.append("Stay")
        elif px - ix == 0 and py - iy == 1:
            actions.append("Left")
        elif px - ix == 0 and py - iy == -1:
            actions.append("Right")
    return actions


def score(state):
    if state.cat_pos == state.mouse_pos:
        return 1
    hole_pos = tuple(np.argwhere(state.grid[HOLE] == 1)[0])
    if state.mouse_pos == hole_pos:
        return -1
    return 0


def getdist(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return max(np.fabs(x1 - x2), np.fabs(y1 - y2))


def children_of(state, r=False):
    children = []
    grid, turn = state.grid, state.turn
    actions = valid_actions(state)
    for action in actions:
        node = get_node(action, grid, turn)
        traps = list(map(tuple, np.argwhere(grid[TRAP] != 0)))
        condition = get_explored_condition(r, node, traps)
        if condition:
            children.append(node)
    return children


def get_explored_condition(r, node, traps):
    if node.mouse_pos in traps:
        return True
    elif r is True and (tuple((node.cat_pos, node.mouse_pos)) not in rollouts_visited.keys()):
        return True
    elif r is False and (tuple((node.cat_pos, node.mouse_pos)) not in explored.keys()):
        return True
    return False


def get_node(action, g, turn):
    grid = g.copy()
    dx, dy = action
    px, py = tuple(np.argwhere(grid[turn] == 1)[0])
    grid[turn][px, py] = 0
    grid[turn][px + dx, py + dy] = 1
    grid[EMPTY][px, py] = 1
    grid[EMPTY][px + dx, py + dy] = 0
    if turn == MOUSE:
        new_node = Node(grid, CAT)
    else:
        new_node = Node(grid, MOUSE)
    return new_node


def make_grid():
    grid = np.zeros((6, SIZE, SIZE), dtype=float)
    grid[WALL][SIZE // 2 - 1:SIZE // 2 + 2, SIZE // 2] = 1
    grid[WALL][SIZE // 2, SIZE // 2 - 1:SIZE // 2 + 2] = 1
    grid[EMPTY][grid[WALL] == 0] = 1
    grid = put_agents(grid)
    return grid


def put_agents(g):
    grid_cat, grid_mouse, grid_traps, grid_hole, grid_empty = g[CAT].flatten(), g[MOUSE].flatten(), g[TRAP].flatten(), \
                                                              g[HOLE].flatten(), g[EMPTY].flatten()
    pos = [i for i in range(len(grid_empty)) if grid_empty[i] == 1]
    m, c, t1, t2, t3, h = np.random.choice(pos, size=6, replace=False)
    grid_mouse[m], grid_cat[c], grid_traps[t1], grid_traps[t2], grid_traps[t3], grid_hole[h] = np.ones(6)
    grid_empty[m], grid_empty[c], grid_empty[t1], grid_empty[t2], grid_empty[t3], grid_empty[h] = np.zeros(6)
    g[EMPTY], g[CAT], g[MOUSE], g[TRAP], g[HOLE] = grid_empty.reshape((SIZE, SIZE)), grid_cat.reshape(
        (SIZE, SIZE)), grid_mouse.reshape((SIZE, SIZE)), grid_traps.reshape((SIZE, SIZE)), grid_hole.reshape(
        (SIZE, SIZE))
    return g


def is_leaf(state):
    return score(state) != 0


def valid_actions(state):
    actions = []
    grid, turn = state.grid, state.turn
    px, py = tuple(np.argwhere(grid[turn] == 1)[0])
    if turn == MOUSE and grid[TRAP][px, py] == 1:
        grid[TRAP][px, py] = -1
        state.grid = grid
        return [(0, 0)]
    if turn == MOUSE and grid[TRAP][px, py] == -1:
        grid[TRAP][px, py] = 1
        state.grid = grid
    if px < SIZE - 1 and grid[WALL][px + 1, py] != 1: actions.append((1, 0))  # Down
    if px > 0 and grid[WALL][px - 1, py] != 1: actions.append((-1, 0))  # Up
    if py < SIZE - 1 and grid[WALL][px, py + 1] != 1: actions.append((0, 1))  # Right
    if py > 0 and grid[WALL][px, py - 1] != 1: actions.append((0, -1))  # Left
    if px > 0 and py < SIZE - 1 and grid[WALL][px - 1, py + 1] != 1: actions.append((-1, 1))  # NE
    if px < SIZE - 1 and py < SIZE - 1 and grid[WALL][px + 1, py + 1] != 1: actions.append((1, 1))  # SE
    if px < SIZE - 1 and py > 0 and grid[WALL][px + 1, py - 1] != 1: actions.append((1, -1))  # SW
    if px > 0 and py > 0 and grid[WALL][px - 1, py - 1] != 1: actions.append((-1, -1))  # NW
    return actions


class Node:

    def __init__(self, grid, turn):
        self.grid = grid  # self.make_grid()
        self.turn = turn  # CAT
        self.cat_pos = tuple(np.argwhere(grid[CAT] == 1)[0])
        self.mouse_pos = tuple(np.argwhere(grid[MOUSE] == 1)[0])
        self.visit_count = 0
        self.score_total = 0
        self.score_estimate = 0
        self.child_list = None

    def children(self, rt=False):
        if self.child_list is None:
            self.child_list = list(children_of(self, rt))
        return self.child_list

    def N_values(self, rt=False):
        return [c.visit_count for c in self.children(rt)]

    def Q_values(self, rt=False):
        children = self.children(rt)
        sign = +1 if self.turn == CAT else -1
        Q = [sign * c.score_total / (c.visit_count + 1) for c in children]
        return Q


def exploit(node):
    child = node.children()[np.argmax(node.Q_values())]
    return child


def nn_exploit(node):
    children = node.children()

    # To change NN, use any of the functions defined in mcst_nn.py (lin_lin_0006(), lin_lin_0001(), lin_0005(), conv_lin_001())
    model = models.lin_lin_0006()
    mchild, m = children[0], 0
    for child in children:
        child.score_estimate = model(tr.tensor(child.grid, dtype=tr.float).reshape(1, 216))[0]
        #child.score_estimate = model(tr.tensor(child.grid, dtype=tr.float).reshape(6, 36))[0]
        if child.score_estimate > m:
            mchild = child

    return mchild


def random_choice(node):
    return np.random.choice(node.children())


def uct(node):
    Q = np.array(node.Q_values(True))
    N = np.array(node.N_values(True))
    U = Q + np.sqrt(np.log(node.visit_count + 1) / (N + 1))
    child = node.children(True)[np.argmax(U)]
    return child


choose_child = uct


def rollout(node, rollout_limit=100, nn=False):
    rollouts_visited[tuple((node.cat_pos, node.mouse_pos))] = node.score_estimate
    if is_leaf(node) or node.children(True) == [] or rollout_limit <= 0:
        result = score(node)
        rollouts.append(100 - rollout_limit)
    elif rollout_limit < 50 and nn:
        rollout_limit -= 1
        result = rollout(nn_exploit(node), rollout_limit)
    else:
        rollout_limit -= 1
        result = rollout(choose_child(node), rollout_limit)
    node.visit_count += 1
    node.score_total += result
    node.score_estimate = node.score_total / node.visit_count
    rollouts_visited[tuple((node.cat_pos, node.mouse_pos))] = node.score_estimate
    return result
