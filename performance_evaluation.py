import main as m
import domain_mcst as dm
import matplotlib.pyplot as plt
import numpy as np
from nn import NeurNet

def perform_experiment():
    performances, nodes = [], []
    for i in range(100):
        print("Playing Game ", i + 1)
        dm.SIZE = 10
        initial_state = dm.Node(dm.make_grid(), dm.CAT)
        for r in range(m.num_rollouts * 10):
            dm.rollouts_visited = {}
            dm.rollout(initial_state)
        ni = sum(dm.rollouts)
        p, n = m.cat_vs_mouse(initial_state, 2)
        performances.append(p)
        nodes.append(ni + n)
    # print("Performances: ", performances)
    # print("Nodes: ", nodes)
    return performances, nodes


def nn_tree():
    performances, nodes = [], []
    dm.SIZE = 6
    for i in range(100):
        print("Playing Game ", i + 1)
        initial_state = dm.Node(dm.make_grid(), dm.CAT)
        p, n = m.cat_vs_mouse(initial_state, 5)
        performances.append(p)
        nodes.append(n)
        # print("Performances: ", performances)
        # print("Nodes: ", nodes)
    return performances, nodes


performance, node_count = nn_tree()

print(performance, node_count)

figure, axis = plt.subplots(1, 2, figsize=(15,15))
figure.suptitle('NN + Tree')

axis[0].hist(node_count,
                bins=np.arange(min(node_count) - 1000, max(node_count) + 1000, sum(node_count) / len(node_count)))
axis[0].set_title('Nodes')
axis[0].set_xlabel('Number of Nodes')
axis[0].set_ylabel('Frequency')

axis[1].hist(performance, bins=np.arange(min(performance) - 1, max(performance) + 1, 0.5))
axis[1].set_title('Performance')
axis[1].set_xlabel('Performance')
axis[1].set_ylabel('Frequency')

figure.subplots_adjust(wspace=0.5, hspace=0.5)
figure.savefig('nntree_6_conv_lin_001.jpg')
