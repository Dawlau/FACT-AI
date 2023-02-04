import os
import networkx as nx
import numpy as np

from argparse import ArgumentParser
from collections import defaultdict
from pyvis.network import Network

# Constants
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT_DIR, 'datahub')
DATASETS = {
    # Soft Self-avoiding random walk
    'soft_synth2': '/soft_synth2/soft_synth2',
    'soft_synth3': '/soft_synth3/soft_synth3',
    'soft_rice_subset': '/soft_rice_subset/soft_rice_subset',

    # Standard random walk algorithm
    'synth2': '/synth2/synth2',
    'synth3': '/synth3/synth3',
    'rice_subset': '/rice_subset/rice_subset',  # takes a lot to render
    # 'synthetic_3layers': '/synthetic_3layers/synthetic_3layers_n500_Pred0.7_Phom0.025_Phet0.001',# color are not right
    # 'twitter': '/twitter/twitter',  # takes a lot to render
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
    num_lines = sum(1 for lines in open(DATA + "/" + dataset + "/" + dataset + '.embeddings_' + embedding + '.walks.0'))
    with open(DATA + "/" + dataset + "/" + dataset + '.embeddings_' + embedding + '.walks.0') as walk_file:
        walks_idx = np.random.choice(num_lines, walks_visualised)
        for i, line in enumerate(walk_file):
            if i in walks_idx:
                walk = line.split()
                walks.append(walk)
    return walks

def get_graph_map(dataset, embedding):
    graph = defaultdict(float)
    filename = DATA + "/" + dataset + "/" + dataset + '.embeddings_' + embedding + '.graph.out'
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
    for edge in G_map.keys():
        color = colour_picker(G_map[(edge[0], edge[1])])
        G.add_edge(edge[0], edge[1],
                   value=1,  # 5*G_map[(edge[0], edge[1])]
                   color=color)  # matplotlib.colors.rgb2hex(color))
    net = Network()
    net.barnes_hut()
    net.from_nx(G)
    net.repulsion()
    net.show_buttons()
    net.show(f"results/{dataset}_{embedding}_edge_vis.html")
    net.toggle_physics(False)


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
    net.show(f"results/{dataset}_{embedding}.html")
    net.toggle_physics(False)

def main(args):
    # dependency https://pypi.org/project/pyvis/: pip install pyvis
    rw_method = f'random_walk_5_bndry_{args.alpha}_exp_{args.p}_d32_1'
    srw_method = f'c{args.c}_random_walk_5_bndry_{args.alpha}_exp_{args.p}_d32_1'

    print('Deepwalk')
    visualize_walks('synth2', 'unweighted_d32_1', 1)
    visualize_walks('synth3', 'unweighted_d32_1', 1)
    visualize_edge_weights('synth2', 'unweighted_d32')
    visualize_edge_weights('synth3', 'unweighted_d32')

    print('CrossWalk using default random walks.')
    visualize_walks('synth2', rw_method, 1)
    visualize_walks('synth3', rw_method, 1)
    visualize_edge_weights('synth2', rw_method)
    visualize_edge_weights('synth3', rw_method)

    print('CrossWalk using Soft Self-avoiding random walks.')
    visualize_walks('soft_synth2', srw_method, 1)
    visualize_walks('soft_synth3', srw_method, 1)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--alpha", default=0.5, help="alpha param")
    parser.add_argument("--p", default=1.0, help="exp param")
    parser.add_argument("--c", default=0.3, help="statistics param")

    args = parser.parse_args()
    main(args)

