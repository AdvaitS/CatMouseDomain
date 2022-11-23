# MCTS Implementation
import numpy as np

EMPTY, CAT, MOUSE, WALL, TRAP, HOLE = [0., 1., 2., -1., 0.5, 0.7]
SIZE = -1


def state_string(grid):
    state = grid
    state[state == str(EMPTY)] = "_"
    state[state == str(CAT)] = "C"
    state[state == str(MOUSE)] = "M"
    state[state == str(WALL)] = "W"
    state[state == str(TRAP)] = "T"
    state[state == str(HOLE)] = "H"
    return "\n".join(["  ".join(row) for row in state])


def score(state):
    for player, value in (("X", 1), ("O", -1)):
        if (state == player).all(axis=0).any(): return value
        if (state == player).all(axis=1).any(): return value
        if (np.diag(state) == player).all(): return value
        if (np.diag(np.rot90(state)) == player).all(): return value
    return 0


def get_player(state):
    return state.turn



def children_of(state):
    # symbol = get_player(state)
    children = []
    grid, turn = state.grid, state.turn
    actions = valid_actions(state)
    for action in actions:
        children.append(perform_action(action, grid, turn))
    return children


def perform_action(action, g, turn):
    grid = g
    dx, dy = action
    if turn == CAT:
        px, py = self.cat_pos
        grid[px, py] -= CAT
        grid[px+dx, py+dy] += CAT
        self.cat_pos = (px+dx, py+dy)
        self.turn = MOUSE
    else:
        px, py = self.mouse_pos
        if grid[px, py] == -(MOUSE+TRAP):
            grid[px, py] = MOUSE+TRAP
        grid[px, py] -= MOUSE
        grid[px+dx, py+dy] += MOUSE
        self.mouse_pos = (px+dx, py+dy)
        self.turn = CAT
    return grid



def get_position(grid, player):
    positions = np.where(grid == player)
    return list(zip(positions[0], positions[1]))[0]

def is_leaf(state):
    children = children_of(state)
    value = score(state)
    return len(children) == 0 or value != 0

def valid_actions(state):
    actions = []
    grid, turn = state.grid, state.turn
    px, py = get_position(grid, turn)
    print(px, py)
    if turn == MOUSE and grid[px, py] == MOUSE+TRAP:
        grid[px, py] *= -1
        return [(0, 0)]
    if px < SIZE - 1 and grid[px+1, py] != WALL: actions.append((1, 0))                             #Down
    if px > 0 and grid[px-1, py] != WALL: actions.append((-1, 0))                                   #Up
    if py < SIZE - 1 and grid[px, py+1] != WALL: actions.append((0, 1))                             #Right
    if py > 0 and grid[px, py-1] != WALL: actions.append((0, -1))                                   #Left
    if px > 0 and py < SIZE - 1 and grid[px-1, py+1] != WALL: actions.append((-1, 1))               #NE
    if px < SIZE - 1 and py < SIZE - 1 and grid[px+1, py+1] != WALL: actions.append((1, 1))         #SE
    if px < SIZE - 1 and py > 0 and grid[px+1, py-1] != WALL: actions.append((1, -1))               #SW
    if px > 0 and py > 0 and grid[px-1, py-1] != WALL: actions.append((-1, -1))                     #NW
    return actions

class Node:

    def __init__(self):
        self.grid = self.make_grid()
        self.turn = CAT
        self.cat_pos = get_position(self.grid, CAT)
        self.mouse_pos = get_position(self.grid, MOUSE)
        self.visit_count = 0
        self.score_total = 0
        self.score_estimate = 0
        self.child_list = None

    def make_grid(self):
        grid = np.zeros((SIZE, SIZE), dtype=float)
        grid[SIZE // 2 - 1:SIZE // 2 + 2, SIZE // 2] = WALL
        grid[SIZE // 2, SIZE // 2 - 1:SIZE // 2 + 2] = WALL
        grid = self.put_agents(grid)
        return grid

    def put_agents(self, g):
        grid_copy = g.flatten()
        pos = [i for i in range(len(grid_copy)) if grid_copy[i] == EMPTY]
        m, c, t, h = np.random.choice(pos, size=4, replace=False)
        grid_copy[m], grid_copy[c], grid_copy[t], grid_copy[h] = MOUSE, CAT, TRAP, HOLE
        return grid_copy.reshape((SIZE, SIZE))

    def children(self):
        # Only generate children the first time they are requested and memoize
        if self.child_list is None:
            self.child_list = list(map(Node, self.children_of()))
        # Return the memoized child list thereafter
        return self.child_list

    # Helper to collect child visit counts into a list
    def N_values(self):
        return [c.visit_count for c in self.children()]

    # Helper to collect child estimated utilities into a list
    # Utilities are from the current player's perspective
    def Q_values(self):
        children = self.children()

        # negate utilities for min player "O"
        sign = +1 if self.turn == CAT else -1

        # empirical average child utilities
        # special case to handle 0 denominator for never-visited children
        Q = [sign * c.score_total / (c.visit_count + 1) for c in children]
        # Q = [sign * c.score_total / max(c.visit_count, 1) for c in children]

        return Q


# exploit strategy: choose the best child for the current player
def exploit(node):
    return node.children()[np.argmax(node.Q_values())]


# explore strategy: choose the least-visited child
def explore(node):
    return node.children()[np.argmin(node.N_values())]  # TODO: replace with exploration


# upper-confidence bound strategy
def uct(node):
    # max_c Qc + sqrt(ln(Np) / Nc)
    Q = np.array(node.Q_values())
    N = np.array(node.N_values())
    U = Q + np.sqrt(np.log(node.visit_count + 1) / (N + 1))  # +1 for 0 edge case
    return node.children()[np.argmax(U)]


# choose_child = exploit
# choose_child = explore
choose_child = uct


def rollout(node):
    if is_leaf(node.state):
        result = score(node.state)
    else:
        result = rollout(choose_child(node))
    node.visit_count += 1
    node.score_total += result
    node.score_estimate = node.score_total / node.visit_count
    return result
