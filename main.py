import numpy as np
import domain_mcst as dm

num_rollouts = 100
"""
SIZE = int(input("Select a grid size: 6, 7, 8, 9, 10\nChoice: "))
dm.SIZE = SIZE
state = dm.Node(dm.make_grid(), dm.CAT)


print("Initial State:")
print(dm.state_string(state.grid))

print("Performing initial rollouts...")
for r in range(num_rollouts*10):
    dm.rollouts_visited = {}
    dm.rollout(state)
print(dm.rollouts)
game_type = int(input("Tree Games:\n1. Optimal Cat vs Optimal Mouse\n2. Optimal Cat vs Sub-Optimal "
                      "Mouse\n3. Sub-Optimal Cat vs Optimal Mouse\n4. Sub-Optimal Cat vs Sub-Optimal Mouse\n5. "
                      "Optimal Cat vs Interactive Mouse\n6. Interactive Cat vs Optimal Mouse\nEnter Choice: "))

"""
def cat_vs_mouse(state, gametype):
    moves = 0
    curr_state = state
    dm.explored = {}
    number_of_nodes = 0
    strategies = {1: (dm.exploit, dm.exploit),
                  2: (dm.exploit, dm.random_choice),
                  3: (dm.random_choice, dm.exploit),
                  4: (dm.random_choice, dm.random_choice)}
    cat_strategy, mouse_strategy = strategies[gametype]
    flag = strategies[gametype] != 4
    while not dm.is_leaf(curr_state) and moves <= 100 and curr_state.children() != []:
        if flag:
            dm.rollouts = []
            for r in range(num_rollouts):
                dm.rollouts_visited = {}
                dm.rollout(curr_state)
        if curr_state.turn == dm.CAT:
            print("Cat's Move")
            curr_state = cat_strategy(curr_state)
        else:
            print("Mouse' Move")
            curr_state = mouse_strategy(curr_state)
        moves += 1
        dm.explored[tuple((curr_state.cat_pos, curr_state.mouse_pos))] = curr_state.score_estimate
        print(dm.state_string(curr_state.grid))
        print(dm.rollouts)
        number_of_nodes += sum(dm.rollouts)
    print("GAME OVER")
    print("Moves: ", moves)
    sign, dist = 0, 0
    if dm.winner(curr_state) == dm.CAT:
        print("Cat Wins")
        sign = 1
        dist = dm.getdist(state.cat_pos, state.mouse_pos)
    elif dm.winner(curr_state) == dm.MOUSE:
        print("Mouse Wins")
        sign = -1
        dist = dm.getdist(state.mouse_pos, tuple(np.argwhere(state.grid[dm.HOLE] == 1)[0]))
    print("Performance: ", (sign * dist) / moves)
    return (sign * dist) / moves, number_of_nodes


def interactive_mouse(state):
    moves = 0
    curr_state = state
    while not dm.is_leaf(curr_state):
        for r in range(num_rollouts):
            dm.rollouts_visited = {}
            dm.rollout(curr_state)
        if curr_state.turn == dm.CAT:
            print("Cat's Move")
            curr_state = dm.exploit(curr_state)
        else:
            print("Mouse' Move")
            print("Valid Actions: ", dm.get_actions(curr_state))
            action = int(input("Enter index of the action: "))
            curr_state = curr_state.children()[action]
        moves += 1
        dm.explored[tuple((curr_state.cat_pos, curr_state.mouse_pos))] = curr_state.score_estimate
        print(dm.state_string(curr_state.grid))
    print("GAME OVER")
    print("Moves: ", moves)
    sign, dist = 0, 0
    if dm.winner(curr_state) == dm.CAT:
        print("Cat Wins")
        sign = 1
        dist = dm.getdist(state.cat_pos, state.mouse_pos)
    elif dm.winner(curr_state) == dm.MOUSE:
        print("Mouse Wins")
        sign = -1
        dist = dm.getdist(state.mouse_pos, tuple(np.argwhere(state.grid[dm.HOLE] == 1)[0]))
    print("Performance: ", (sign * dist) / moves)


def interactive_cat(state):
    moves = 0
    curr_state = state
    while not dm.is_leaf(curr_state):
        for r in range(num_rollouts):
            dm.rollouts_visited = {}
            dm.rollout(curr_state)
        if curr_state.turn == dm.CAT:
            print("Cat's Move")
            print("Valid Actions: ", dm.get_actions(curr_state))
            action = int(input("Enter index of the action: "))
            curr_state = curr_state.children()[action]
        else:
            print("Mouse' Move")
            curr_state = dm.exploit(curr_state)
        moves += 1
        dm.explored[tuple((curr_state.cat_pos, curr_state.mouse_pos))] = curr_state.score_estimate
        print(dm.state_string(curr_state.grid))
    print("GAME OVER")
    print("Moves: ", moves)
    sign, dist = 0, 0
    if dm.winner(curr_state) == dm.CAT:
        print("Cat Wins")
        sign = 1
        dist = dm.getdist(state.cat_pos, state.mouse_pos)
    elif dm.winner(curr_state) == dm.MOUSE:
        print("Mouse Wins")
        sign = -1
        dist = dm.getdist(state.mouse_pos, tuple(np.argwhere(state.grid[dm.HOLE] == 1)[0]))
    print("Performance: ", (sign * dist) / moves)

"""
if game_type in [1, 2, 3, 4]:
    cat_vs_mouse(state, game_type)
elif game_type == 5:
    interactive_mouse(state)
elif game_type == 6:
    interactive_cat(state)
else:
    print("Invalid Input")
"""