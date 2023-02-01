import json
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict

# Constants
DATA = '../datahub'
WALK_LENGTH = 40
DATASETS = {
    # Soft Self-avoiding random walk
    'soft_synth2': '/soft_synth2/synthetic_n500_Pred0.7_Phom0.025_Phet0.001',
    'soft_synth3': '/soft_synth3/synthetic_3g_n500_Pred0.6_Pblue0.25_Prr0.025_Pbb0.025_Pgg0.025_Prb0.001_Prg0.0005_Pbg0.0005',
    'soft_rice_subset': '/soft_rice_subset/soft_rice_subset',

    # Standard random walk algorithm
    'synth2': '/synth2/synthetic_n500_Pred0.7_Phom0.025_Phet0.001',
    'synth3': '/synth3/synthetic_3g_n500_Pred0.6_Pblue0.25_Prr0.025_Pbb0.025_Pgg0.025_Prb0.001_Prg0.0005_Pbg0.0005',
    'rice_subset': '/rice_subset/rice_subset',  # takes a lot to render
    # 'synthetic_3layers': '/synthetic_3layers/synthetic_3layers_n500_Pred0.7_Phom0.025_Phet0.001',# color are not right
    # 'twitter': '/twitter/twitter',  # takes a lot to render
}


def parse_walks(dataset, embedding):
    walks = []
    filename = DATA + "/" + dataset + "/" + dataset + '.embeddings_' + embedding + '.walks.0'
    with open(filename, 'r') as walk_file:
        for line in walk_file:
            walks.append([int(node_id) for node_id in line.split()])
    return walks


def parse_graph(dataset, embedding):
    graph = defaultdict(float)
    filename = DATA + "/" + dataset + "/" + dataset + '.embeddings_' + embedding + '.graph.out'
    with open(filename, 'r') as graph_file:
        for line in graph_file:
            from_, to_, weight = line.split()
            graph[(from_, to_)] = float(weight)
    return graph


def parse_node_properties(dataset):
    property_map = dict()
    with open(DATA + DATASETS[dataset] + '.attr', 'r') as attr_file:
        for line in attr_file:
            node_id, attr = line.split()
            property_map[int(node_id)] = attr
    return property_map


def plot_edges_histogram(dataset, embedding):
    G = parse_graph(dataset, embedding)
    edge_weights = np.array(list(G.values()))
    hist_array, bins = np.histogram(edge_weights, bins=25)
    plt.hist(bins[:-1], bins, weights=hist_array)
    plt.title(f'Edge weights distribution: {dataset} - {embedding}')
    plt.show()


def generate_rw_statistics(dataset, embedding):
    walks = parse_walks(dataset, embedding)
    property_map = parse_node_properties(dataset)
    nodes_visited = []
    num_boundary_crossed = []
    for walk in walks:
        nodes_visited.append(len(np.unique(walk)))
        crossings = 0
        # when going from a to b, if prop_a != prop_b: boundary was crossed.
        for i in range(len(walk) - 2):
            crossing_condition = property_map[walk[i]] != property_map[walk[i + 1]]
            crossings += 1 if crossing_condition else 0
        num_boundary_crossed.append(crossings)

    with open(f'{dataset}_{embedding}_stats.json', 'w') as out_file:
        result = {'walk_length': WALK_LENGTH,
                  'avg_visited': np.round(np.mean(nodes_visited), 2),
                  'avg_crossings': np.round(np.mean(num_boundary_crossed), 2)}
        json.dump(result, out_file)
        print(f'  Dataset: {dataset} - Embedding: {embedding}')
        print(f'  Average number of nodes visited: {result["avg_visited"]} '
              f'({np.round(100 * result["avg_visited"] / WALK_LENGTH, 2)}% of walk explored)')
        print(f'  Average number of times boundary was crossed: {result["avg_crossings"]}\n')

def main(args):
    rw_method = f'random_walk_5_bndry_{args.alpha}_exp_{args.p}_d32_1'
    srw_method = f'c{args.c}_random_walk_5_bndry_{args.alpha}_exp_{args.p}_d32_1'

    print('Deepwalk')
    generate_rw_statistics('synth2', 'unweighted_d32_1')
    generate_rw_statistics('synth3', 'unweighted_d32_1')
    generate_rw_statistics('rice_subset', 'unweighted_d32_1')

    print('CrossWalk using default random walks.')
    generate_rw_statistics('synth2', rw_method)
    generate_rw_statistics('synth3', rw_method)
    generate_rw_statistics('rice_subset', rw_method)

    print('CrossWalk using Soft Self-avoiding random walks.')
    generate_rw_statistics('soft_synth2', srw_method)
    generate_rw_statistics('soft_synth3', srw_method)
    generate_rw_statistics('soft_rice_subset', srw_method)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--alpha", default=0.5, help="alpha param")
    parser.add_argument("--p", default=1.0, help="exp param")
    parser.add_argument("--c", default=0.35, help="regularization param")

    args = parser.parse_args()
    main(args)
