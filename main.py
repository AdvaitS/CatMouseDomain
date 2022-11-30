import numpy as np
import domain_mcst as dm

SIZE = int(input("Select a grid size: 5, 7, 10, 12, 15\nChoice: "))
dm.SIZE = SIZE

state = dm.Node(dm.make_grid(), dm.CAT)
print("Initial State:")
print(dm.state_string(state.grid.astype(str)))
# print("Starting Grid: \n", dm.state_string(state.grid.astype(str)))
# print(state.grid)
# dm.valid_actions(state)
# gauge sub-optimality with rollouts
num_rollouts = 100
# node = dm.Node(grid)
#for r in range(num_rollouts):
#    dm.explored = set()
#    dm.rollout(state)
#    if r % (num_rollouts // 10) == 0: print(r, state.score_estimate)


def mcst_AI(state):
    moves = 0
    while not dm.is_leaf(state) or moves <= 100:
        print(moves)
        for r in range(num_rollouts):
            dm.explored = set()
            dm.rollout(state)
        if state.turn == dm.CAT:
            print("Cat's Move")
            state = dm.exploit(state)
        else:
            print("Mouse' Move")
            state = dm.exploit(state)
        moves += 1
        print(dm.state_string(state.grid.astype(str)))
    print("GAME OVER")
    sign = 0
    if dm.winner(state) == dm.CAT:
        sign = 1
    elif dm.winner(state) == dm.MOUSE:
        sign = -1
    print("Performance: ", 1/moves * sign)

def baseline_AI(state):
    while not dm.is_leaf(state):
        if state.turn == dm.CAT:
            print("Cat's Move")
            state = np.random.choice(state.children())
        else:
            print("Mouse' Move")
            state = np.random.choice(state.children())
        print(dm.state_string(state.grid.astype(str)))
    print("GAME OVER")
    print("Score: ", state.score_estimate)

def interactive(state):
    while not dm.is_leaf(state):
        if state.turn == dm.CAT:
            print("Cat's Move")
            state = dm.exploit(state)
        else:
            print("Mouse' Move")
            actions = state.children()
            print("Choose an action: ", [(i.mouse_pos) for i in actions])
            action = int(input())
            state = state.children()[action]
        print(dm.state_string(state.grid.astype(str)))
    print("GAME OVER")
    print("Score: ", state.score_estimate)

#interactive(state)
mcst_AI(state)