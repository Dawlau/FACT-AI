from collections import defaultdict

import matplotlib
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from pyvis.network import Network

# Constants
DATA = '../data'
DATASETS = {
    'synth2': '/synth2/synthetic_n500_Pred0.7_Phom0.025_Phet0.001',
    # 'rice_subset': '/rice_subset/rice_subset',  # takes a lot to render
    # 'synthetic_3layers': '/synthetic_3layers/synthetic_3layers_n500_Pred0.7_Phom0.025_Phet0.001',# color are not right
    # 'twitter': '/twitter/twitter',  # takes a lot to render
    # 'synth3': '/synth3/synthetic_3g_n500_Pred0.6_Pblue0.25_Prr0.025_Pbb0.025_Pgg0.025_Prb0.001_Prg0.0005_Pbg0.0005',
}

def parse_graph(dataset):
    G = nx.read_edgelist(DATA + DATASETS[dataset] + '.links')
    # Update nodes with the properties attr
    with open(DATA + DATASETS[dataset] + '.attr', 'r') as atr_file:
        for line in atr_file:
            node_attr = line.split()  # (id, group, opt[extra])
            if node_attr[0] in G.nodes:
                G.nodes[node_attr[0]]['group'] = node_attr[1]
    return G

def parse_walks(dataset, embedding, walks_visualised):
    # randomly sampling walks
    walks = []
    num_lines = sum(1 for lines in open(DATA + DATASETS[dataset] + '.embeddings_' + embedding + '.walks.0'))
    with open(DATA + DATASETS[dataset] + '.embeddings_' + embedding + '.walks.0') as walk_file:
        walks_idx = np.random.choice(num_lines, walks_visualised)
        for i, line in enumerate(walk_file):
            if i in walks_idx:
                walk = line.split()
                walks.append(walk)
    return walks

def get_graph_map(dataset, embedding):
    graph = defaultdict(float)
    filename = DATA + DATASETS[dataset] + '.embeddings_' + embedding + '.graph.out'
    with open(filename, 'r') as graph_file:
        for line in graph_file:
            from_, to_, weight = line.split()
            graph[(from_, to_)] = float(weight)
    return graph

def colour_picker(value):
    if value <= 0.1:
        return '#481769'
    elif value <= 0.4:
        return '#287a8e'
    elif value <= 0.6:
        return '#77d152'
    else:
        return '#fce724'

def visualize_edge_weights(dataset, embedding):
    G = parse_graph(dataset)
    G_map = get_graph_map(dataset, embedding)
    cmap = plt.cm.viridis
    edge_weights = list(G_map.values())
    for edge in G_map.keys():
        # color = cmap(G_map[(edge[0], edge[1])]/max(edge_weights))
        color = colour_picker(G_map[(edge[0], edge[1])])
        G.add_edge(edge[0], edge[1],
                   value=1,  # 5*G_map[(edge[0], edge[1])]
                   color=color)  # matplotlib.colors.rgb2hex(color))
    net = Network()
    net.barnes_hut()
    net.from_nx(G)
    net.repulsion()
    net.show_buttons()
    # net.toggle_physics(False)
    net.show(f"{dataset}_{embedding}_edge_vis.html")



def visualize_walks(dataset, embedding, walks_visualised):
    """
    This function visualizes the random walks on the graph.
    """
    G = parse_graph(dataset)
    walks = parse_walks(dataset, embedding, walks_visualised)
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
    net.show_buttons()
    net.show(f"{dataset}_{embedding}.html")


if __name__ == '__main__':
    # dependency https://pypi.org/project/pyvis/: pip install pyvis
    # run this while in directory /visualisation

    for dataset in DATASETS:
        visualize_walks(dataset, 'unweighted_d32', 1)
        visualize_walks(dataset, 'random_walk_5_bndry_0.5_exp_2.0_d32', 1)
        visualize_walks(dataset, 'random_walk_5_bndry_0.5_exp_4.0_d32', 1)

        visualize_edge_weights(dataset, 'unweighted_d32')
        visualize_edge_weights(dataset, 'random_walk_5_bndry_0.5_exp_2.0_d32')
        visualize_edge_weights(dataset, 'random_walk_5_bndry_0.5_exp_4.0_d32')



