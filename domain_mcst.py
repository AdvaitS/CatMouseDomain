# MCTS Implementation
import numpy as np

EMPTY, CAT, MOUSE, WALL, TRAP, HOLE = [0., 1., 2., -1., 0.5, 0.7]
SIZE = -1
explored = set()

def state_string(grid):
    state = grid
    state[state == str(EMPTY)] = "_"
    state[state == str(CAT)] = "C"
    state[state == str(MOUSE)] = "M"
    state[state == str(WALL)] = "W"
    state[state == str(TRAP)] = "T"
    state[state == str(HOLE)] = "H"
    state[state == str(CAT+MOUSE)] = "CM"
    state[state == str(CAT+TRAP)] = "CT"
    state[state == str(CAT+HOLE)] = "CH"
    state[state == str(MOUSE + TRAP)] = "MT"
    state[state == str(MOUSE + HOLE)] = "MH"
    state[state == str(CAT+MOUSE+HOLE)] = "CMH"
    state[state == str(CAT+MOUSE+TRAP)] = "CMT"
    return "\n".join(["  ".join(row) for row in state])


def score(state):
    """
    grid, turn = state.grid, state.turn
    if HOLE in grid:
        hole_position = np.where(grid == HOLE)
    elif MOUSE + HOLE in grid:
        hole_position = np.where(grid == MOUSE+HOLE)
    else:
        hole_position = np.where(grid == MOUSE+TRAP)
    hposition = list(zip(hole_position[0], hole_position[1]))[0]
    return getdist(state.mouse_pos, hposition) - getdist(state.cat_pos, state.mouse_pos)
    """
    if MOUSE+CAT in state.grid:
        return 1
    if MOUSE+HOLE in state.grid:
        return -1
    return 0


def getdist(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return max(np.fabs(x1 - x2), np.fabs(y1 - y2))


def get_player(state):
    return state.turn


def children_of(state):
    # symbol = get_player(state)
    children = []
    grid, turn = state.grid, state.turn
    actions = valid_actions(state)
    #print("Valid Actions: ", actions)
    for action in actions:
        new_state = perform_action(action, grid, turn)
        if tuple((new_state.cat_pos, new_state.mouse_pos)) not in explored:
            children.append(new_state)
            explored.add(tuple((new_state.cat_pos, new_state.mouse_pos)))
    return children


def perform_action(action, g, turn):
    grid = g.copy()
    dx, dy = action
    new_node = Node(grid, turn)
    #print("Performing action ({0}, {1}) for {2}".format(dx, dy, turn))
    if turn == CAT:
        px, py = new_node.cat_pos
        new_node.grid[px, py] -= CAT
        new_node.grid[px + dx, py + dy] += CAT
        new_node.cat_pos = (px + dx, py + dy)
        new_node.turn = MOUSE
    else:
        px, py = new_node.mouse_pos
        if new_node.grid[px, py] == -(MOUSE + TRAP):
            new_node.grid[px, py] = MOUSE + TRAP
        new_node.grid[px, py] -= MOUSE
        new_node.grid[px + dx, py + dy] += MOUSE
        new_node.mouse_pos = (px + dx, py + dy)
        new_node.turn = CAT
    #print(new_node.grid)
    return new_node


def make_grid():
    grid = np.zeros((SIZE, SIZE), dtype=float)
    grid[SIZE // 2 - 1:SIZE // 2 + 2, SIZE // 2] = WALL
    grid[SIZE // 2, SIZE // 2 - 1:SIZE // 2 + 2] = WALL
    grid = put_agents(grid)
    return grid


def put_agents(g):
    grid_copy = g.flatten()
    pos = [i for i in range(len(grid_copy)) if grid_copy[i] == EMPTY]
    m, c, t, h = np.random.choice(pos, size=4, replace=False)
    grid_copy[m], grid_copy[c], grid_copy[t], grid_copy[h] = MOUSE, CAT, TRAP, HOLE
    return grid_copy.reshape((SIZE, SIZE))


def get_position(grid, player):
    if player == CAT:
        if CAT in grid:
            positions = np.where(grid == CAT)
        elif CAT + HOLE in grid:
            positions = np.where(grid == CAT + HOLE)
        elif CAT + TRAP in grid:
            positions = np.where(grid == CAT + TRAP)
        elif CAT + MOUSE + TRAP in grid:
            positions = np.where(grid == CAT + MOUSE + TRAP)
        else:
            positions = np.where(grid == CAT + MOUSE + HOLE)
    else:
        if MOUSE in grid:
            positions = np.where(grid == MOUSE)
        elif MOUSE + HOLE in grid:
            positions = np.where(grid == MOUSE + HOLE)
        elif MOUSE+TRAP in grid:
            positions = np.where(grid == MOUSE + TRAP)
        elif -(MOUSE+TRAP) in grid:
            positions = np.where(grid == -(MOUSE + TRAP))
        elif CAT + MOUSE + TRAP in grid:
            positions = np.where(grid == CAT + MOUSE + TRAP)
        else:
            positions = np.where(grid == CAT + MOUSE + HOLE)
    #print("Position of {0}: {1}".format(player, positions))
    return list(zip(positions[0], positions[1]))[0]  # np.array([x],[y]) -> (x, y)


def is_leaf(state):
    #children = children_of(state)
    #value = score(state)
    return score(state) != 0


def valid_actions(state):
    actions = []
    grid, turn = state.grid, state.turn
    px, py = get_position(grid, turn)
    #print(px, py)
    if turn == MOUSE and grid[px, py] == MOUSE + TRAP:
        grid[px, py] *= -1
        return [(0, 0)]
    if px < SIZE - 1 and grid[px + 1, py] != WALL: actions.append((1, 0))  # Down
    if px > 0 and grid[px - 1, py] != WALL: actions.append((-1, 0))  # Up
    if py < SIZE - 1 and grid[px, py + 1] != WALL: actions.append((0, 1))  # Right
    if py > 0 and grid[px, py - 1] != WALL: actions.append((0, -1))  # Left
    if px > 0 and py < SIZE - 1 and grid[px - 1, py + 1] != WALL: actions.append((-1, 1))  # NE
    if px < SIZE - 1 and py < SIZE - 1 and grid[px + 1, py + 1] != WALL: actions.append((1, 1))  # SE
    if px < SIZE - 1 and py > 0 and grid[px + 1, py - 1] != WALL: actions.append((1, -1))  # SW
    if px > 0 and py > 0 and grid[px - 1, py - 1] != WALL: actions.append((-1, -1))  # NW
    return actions


class Node:

    def __init__(self, grid, turn):
        self.grid = grid  # self.make_grid()
        self.turn = turn  # CAT
        self.cat_pos = get_position(self.grid, CAT)
        self.mouse_pos = get_position(self.grid, MOUSE)
        self.visit_count = 0
        self.score_total = 0
        self.score_estimate = 0
        self.child_list = None

    def children(self):
        # Only generate children the first time they are requested and memoize
        if self.child_list is None:
            self.child_list = list(children_of(self))
        # Return the memoized child list thereafter
        return self.child_list #[child for child in self.child_list if tuple((child.cat_pos, child.mouse_pos)) not in explored]  #node.children()[np.argmax(U)]

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
    #print("UCT:\n", node.children()[np.argmax(U)].grid)

    return node.children()[np.argmax(U)]


# choose_child = exploit
# choose_child = explore
choose_child = uct


def rollout(node):
    if is_leaf(node) or node.children() == []:
        result = score(node)
    else:
        result = rollout(choose_child(node))
    node.visit_count += 1
    node.score_total += result
    node.score_estimate = node.score_total / node.visit_count
    return result
