import networkx as nx
import numpy as np
from pyvis.network import Network

# Constants
DATA = '../data'
DATASETS = {
    # 'rice_subset': '/rice_subset/rice_subset',  # takes a lot to render
    'synth2': '/synth2/synthetic_n500_Pred0.7_Phom0.025_Phet0.001',
    # 'synthetic_3layers': '/synthetic_3layers/synthetic_3layers_n500_Pred0.7_Phom0.025_Phet0.001',# color are not right
    # 'twitter': '/twitter/twitter',  # takes a lot to render
    # 'synth3': '/synth3/synthetic_3g_n500_Pred0.6_Pblue0.25_Prr0.025_Pbb0.025_Pgg0.025_Prb0.001_Prg0.0005_Pbg0.0005',
}


def visualize_walks(dataset, embedding, walks_visualised):
    """
    This function visualizes the random walks on the graph.
    """
    G = nx.read_edgelist(DATA + DATASETS[dataset] + '.links')
    # Update nodes with the properties attr
    with open(DATA + DATASETS[dataset] + '.attr', 'r') as atr_file:
        for line in atr_file:
            node_attr = line.split()  # (id, group, opt[extra])
            if node_attr[0] in G.nodes:
                G.nodes[node_attr[0]]['group'] = node_attr[1]

    # randomly sampling walks
    walks = []
    num_lines = sum(1 for lines in open(DATA + DATASETS[dataset] + '.embeddings_' + embedding + '.walks.0'))
    with open(DATA + DATASETS[dataset] + '.embeddings_' + embedding + '.walks.0') as walk_file:
        walks_idx = np.random.choice(num_lines, walks_visualised)
        for i, line in enumerate(walk_file):
            if i in walks_idx:
                walk = line.split()
                walks.append(walk)

    # colouring the path, source and target nodes
    for vis, walk in enumerate(walks):
        G.add_node(walk[0], value=500, group='source')
        G.add_node(walk[len(walk) - 2], value=500, group='target')
        for i in range(len(walk)-1):
            G.add_edge(walk[i], walk[i+1], value=500, color='blue')

    # create pyvis object
    net = Network()
    net.barnes_hut()
    net.from_nx(G)
    net.repulsion()

    # draw random walks on the graph.
    # net.show_buttons()
    # net.toggle_physics(False)
    net.show(f"{dataset}_{embedding}.html")


if __name__ == '__main__':
    for dataset in DATASETS:
        visualize_walks(dataset, 'unweighted_d32', 1)
        visualize_walks(dataset, 'random_walk_5_bndry_0.5_exp_2.0_d32', 1)
        visualize_walks(dataset, 'random_walk_5_bndry_0.5_exp_4.0_d32', 1)
