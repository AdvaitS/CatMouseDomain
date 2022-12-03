import main as m
import domain_mcst as dm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter


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


performances, nodes = perform_experiment()
print(performances, nodes)

plt.plot(np.arange(1, 101), nodes)
plt.xticks(np.arange(1, 101, 10))
plt.xlabel('Game')
plt.ylabel('No. of Nodes')
plt.title('Node Count Graph for grid size = 6')
plt.savefig('nodes6.jpg')
plt.show()

plt.plot(np.arange(1, 101), nodes)
plt.xticks(np.arange(1, 101, 10))
plt.xlabel('Game')
plt.ylabel('Performance')
plt.title('Performance Graph for grid size = 6')
plt.savefig('performance6.jpg')
plt.show()
