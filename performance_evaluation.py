import main as m
import domain_mcst as dm
import matplotlib.pyplot as plt
import numpy as np


def perform_experiment():
    performances, nodes = [], []
    for i in range(100):
        print("Playing Game ", i + 1)
        dm.SIZE = 6
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


performance, node_count = perform_experiment()

print(performance, node_count)

figure, axis = plt.subplots(2, 2)
figure.suptitle('size = 6')

axis[0, 0].plot(np.arange(1, 101), node_count)
axis[0, 0].set_xticks(np.arange(1, 101, 1))
axis[0, 0].set_title('Nodes')


axis[0, 1].plot(np.arange(1, 101), performance)
axis[0, 1].set_xticks(np.arange(1, 101, 1))
axis[0, 1].set_title('Performance')


axis[1, 0].hist(node_count, bins=np.arange(min(node_count) - 100, max(node_count) + 100, sum(node_count) / len(node_count)))
axis[1, 0].set_title('Nodes')


axis[1, 1].hist(performance, bins=np.arange(min(performance) - 1, max(performance) + 1, 0.5))
axis[1, 1].set_title('Performance')
figure.subplots_adjust(wspace=0.5, hspace=0.5)
figure.savefig('6.jpg')
