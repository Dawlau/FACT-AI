''' File for testing different files in parallel
'''

from config import infMaxConfig
from generalGreedy import *
import utils as ut
from IC import *
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import os
import time


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def dfs(v, mark, G, colors, num_labels):
    res = np.zeros(num_labels)
    res[int(colors[v])] += 1
    mark.update([v])
    for u in G[v]:
        if u not in mark:
            res += dfs(u, mark, G, colors, num_labels)
    return res


class fairInfMaximization(infMaxConfig):

    def __init__(self, num=-1, args=None):
        super(fairInfMaximization, self).__init__(args)
        self.G = ut.get_data(self.filename, self.weight)


    def test_greedy(self, filename, budget, G_greedy=None):
        generalGreedy_node_parallel(filename, self.G, budget=budget, gamma=None, G_greedy=G_greedy)


    def test_kmedoids(self, emb_filename, res_filename, budget):
        # stats = ut.graph_stats(self.G, print_stats=False)
        v, em = ut.load_embeddings(emb_filename, self.G.nodes())

        influenced, influenced_grouped = [], []
        seeds = []
        for k in range(1, budget + 1):
            print('--------', k)
            S = ut.get_kmedoids_centers(em, k, v)

            I, I_grouped = map_fair_IC((self.G, S))
            influenced.append(I)
            influenced_grouped.append(I_grouped)

            S_g = {c:[] for c in np.unique([self.G.nodes[v]['color'] for v in self.G.nodes])}
            for n in S:
                c = self.G.nodes[n]['color']
                S_g[c].append(n)

            seeds.append(S_g)  # id's of the seeds so the influence can be recreated

        ut.write_files(res_filename, influenced, influenced_grouped, seeds)


def get_walking_method(args):
    walking_algorithm = args.walking_algorithm

    if walking_algorithm == "fairwalk" or walking_algorithm == "unweighted":
        return walking_algorithm
    elif walking_algorithm == "random_walk":
        alpha = args.alpha
        p = args.exponent_p
        return f"random_walk_5_bndry_{alpha}_exp_{p}"
    else:
        if not args.method == "greedy":
            raise Exception("Invalid walking algorithm")


if __name__ == '__main__':

    parser = ArgumentParser("Influence maximization",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')

    parser.add_argument('--method', type=str, help='The method for finding the most influential nodes')
    parser.add_argument('--dataset', type=str, help='The dataset for the current experiment')
    parser.add_argument('--walking_algorithm', type=str, help='The walking algorithm used', default="")
    parser.add_argument('--alpha', type=float, help='The alpha coefficient in the Crosswalk algorithm', nargs='?', default=0)
    parser.add_argument('--exponent_p', type=float, help='The exponent p used in the Crosswalk algorithm', nargs='?', default=0)
    parser.add_argument("--budget", type=int, help="The number of influencial nodes to pick", default=40)

    args = parser.parse_args()

    start = time.time()

    fair_inf = fairInfMaximization(args=args)
    method = args.method
    dataset = args.dataset
    budget = args.budget

    if not (method == "kmedoids" or method == "greedy"):
        raise Exception("Invalid influence maximization algorithm provided")

    for i in range(1, 6):
        if method == "kmedoids":
            embeddings_filename_path = os.path.join(ROOT_DIR, "data", dataset, f"{dataset}.embeddings_{get_walking_method(args)}_d32_{str(i)}")
            results_filename = os.path.join("results", dataset, f"{get_walking_method(args)}_{str(i)}")

            if os.path.exists(f"{results_filename}_results.txt"):
                continue

            fair_inf.test_kmedoids(embeddings_filename_path, results_filename, budget=budget)
        else:
            results_filename = os.path.join("results", dataset, str(i))

            if os.path.exists(f"{results_filename}_greedy__results.txt"):
                continue

            fair_inf.test_greedy(results_filename, budget=budget)

    print('Total time:', time.time() - start)