import numpy as np
import torch as tr
import domain_mcst as dm
import matplotlib.pyplot as pt


class NeurNet(tr.nn.Module):
    def __init__(self, size, hid_features):
        super(NeurNet, self).__init__()
        self.to_hidden = tr.nn.Conv2d(6 * size ** 2, hid_features, kernel_size=(3, 3))
        self.to_output = tr.nn.Linear(hid_features, 1)

    def forward(self, x):
        h = tr.relu(self.to_hidden(x.reshape(x.shape[0], -1)))
        y = tr.tanh(self.to_output(h))
        return y


def generate_data(board_size, num_games):
    dm.SIZE = board_size
    states, results = [], []
    for game in range(num_games):
        print("Game ", game + 1)
        moves = 0
        initial_state = dm.Node(dm.make_grid(), dm.CAT)
        curr_state = initial_state
        for r in range(1000):
            dm.rollouts_visited = {}
            dm.rollout(curr_state)
        while not dm.is_leaf(curr_state) and moves <= 100 and curr_state.children() != []:
            for r in range(100):
                dm.rollouts_visited = {}
                dm.rollout(curr_state)
            for child in curr_state.children():
                if child.score_estimate != 0.0:
                    states.append(child.grid)
                    results.append(child.score_estimate)
            curr_state = dm.random_choice(curr_state)
    return tuple((tr.tensor(np.array(states), dtype=tr.float), tr.tensor(np.array(results), dtype=tr.float)))


def augment(all_states, results):
    augmented_grids = []
    augmented_scores = []
    for STATE, RESULT in zip(all_states, results):
        for k in range(4):
            rot = tr.rot90(STATE, k)
            augmented_grids.append(rot)
            augmented_grids.append(tr.fliplr(rot))
        augmented_scores += [RESULT] * 8
    return tuple((augmented_grids, augmented_scores))


training_examples = generate_data(board_size=6, num_games=10)
testing_examples = generate_data(board_size=6, num_games=2)

# print(len(training_examples[0]))
# print(len(testing_examples[0]))

training_examples = augment(training_examples[0], training_examples[1])
testing_examples = augment(testing_examples[0], testing_examples[1])

print(f"Training: {len(training_examples[0])}\nOutputs: {len(training_examples[1])}")
print(f"Testing: {len(testing_examples[0])}\nOutputs: {len(testing_examples[1])}")

_, utilities = testing_examples
baseline_error = sum((u - 0) ** 2 for u in utilities) / len(utilities)


def example_error(nn, eg):
    state, utility = eg
    x = state.unsqueeze(0)
    y = nn(x)
    err = (y - utility) ** 2
    return err


# Calculates the error on a batch of training examples
def batch_error(nn, batch):
    states_all, utilities_all = batch
    u = utilities_all.reshape(-1, 1).float()
    y = nn(states_all)
    err = tr.sum((y - u) ** 2) / utilities_all.shape[0]
    return err


device = "cuda" if tr.cuda.is_available() else "cpu"
print(device)
# Make the network and optimizer
# net = NeurNet(size=6, hid_features=32)
net = tr.nn.Sequential(
    tr.nn.Conv2d(6, 10, 3),
    tr.nn.ReLU(),
    tr.nn.Flatten(),
    tr.nn.Linear(160, 1),
).to(device)

optimizer = tr.optim.SGD(net.parameters(), lr=0.001)

# Convert the states and their minimax utilities to tensors
states, utilities = training_examples
training_batch = tr.stack(tuple(states)), tr.tensor(utilities)

states, utilities = testing_examples
testing_batch = tr.stack(tuple(states)), tr.tensor(utilities)

# Run the gradient descent iterations
curves = [], []
for epoch in range(50000):

    # zero out the gradients for the next backward pass
    optimizer.zero_grad()

    # loop version (slow)

    e = batch_error(net, training_batch)
    e.backward()
    training_error = e.item()

    with tr.no_grad():
        e = batch_error(net, testing_batch)
        testing_error = e.item()

    # take the next optimization step
    optimizer.step()

    # print/save training progress
    if epoch % 1000 == 0:
        print("%d: %f, %f" % (epoch, training_error, testing_error))
    curves[0].append(training_error)
    curves[1].append(testing_error)


tr.save(net.state_dict(), 'data.txt')
pt.plot(curves[0], 'b-')
pt.plot(curves[1], 'r-')
pt.plot([0, len(curves[1])], [baseline_error, baseline_error], 'g-')
pt.plot()
pt.legend(["Train", "Test", "Baseline"])
pt.show()


"""
-> Replace initial 1000 rollouts with nn
-> in rollouts() when rollout_limit is reached, instead of returning 0, get nn

"""
