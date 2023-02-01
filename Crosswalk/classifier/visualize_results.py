from collections import defaultdict

import numpy as np
import networkx as nx
import math
from os import listdir
import json
import re
from os.path import isfile, join
import os

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

purple_ = "#BF55EC"
yellow_ = "#F7CA18"
label_size = 27
font_size = 27
image_size = (12, 8.5)
line_width = 3
square = True
bar_width = 0.5
legend_size = 25
y_lim = [0, 175]

DATASETS = ["rice_subset" "soft_rice_subset"]


def get_bar_plot(unweighted, fairwalk, rw_method, output_filename):

    xu = [2 - bar_width, 2]
    xf = [5.5 - bar_width, 5.5]
    xp = [9 - bar_width, 9]

    fig, ax = plt.subplots()

    ax.bar(xu[0], unweighted[0], bar_width, color = purple_, edgecolor="black", label="Total Accuracy")
    ax.bar(xf[0], fairwalk[0], bar_width, color = purple_, edgecolor="black")
    ax.bar(xp[0], rw_method[0], bar_width, color = purple_, edgecolor="black")

    ax.bar(xu[1], unweighted[1], bar_width, color = yellow_, edgecolor="black", label="Disparity")
    ax.bar(xf[1], fairwalk[1], bar_width, color = yellow_, edgecolor="black")
    ax.bar(xp[1], rw_method[1], bar_width, color = yellow_, edgecolor="black")

    # ax.set_ylim(y_lim)
    plt.legend(loc="upper right", prop={"size": legend_size})

    plt.xticks([2, 5.5, 9], ["DeepWalk", "FairWalk", "CrossWalk"], fontsize=legend_size)

    # ax.set_axisbelow(True)
    ax.yaxis.grid(color="gray", linestyle="dashed")

    plt.rcParams.update({"font.size": font_size})
    plt.yticks(fontsize=label_size)
    fig.set_size_inches(image_size[0], image_size[1])

    fig.savefig(output_filename, bbox_inches="tight")
    plt.close()


def read_results_txt(filename):
    with open(filename, "r") as r:
        # acc, _, _, _, var = r.readline().strip("(").strip(")").split(",")
        results = json.load(r)
        return [float(results["total_acc"]), float(results["disparity"])]

def plot_pareto_frontier(dataset, maxX=True, maxY=True):
    all_files = [join("results", file) for file in listdir("results") if isfile(join("results", file))]
    dataset_results_files = [file for file in all_files if dataset in file]
    results = defaultdict(lambda: [[], []])

    for file in dataset_results_files:
        with open(file, 'r') as result_file:
            result = json.load(result_file)
            if 'soft' in file:
                results['ssa'][0].append(result['total_acc'])
                results['ssa'][1].append(result['disparity'])
            else:
                results['default'][0].append(result['total_acc'])
                results['default'][1].append(result['disparity'])

    '''Pareto frontier selection process'''
    for method in results.keys():
        sorted_list = sorted([[results[method][1][i], results[method][0][i]]
                              for i in range(len(results[method][0]))], reverse=maxX)
        pareto_front = [sorted_list[0]]
        for pair in sorted_list[1:]:
            if maxY:
                if pair[1] >= pareto_front[-1][1]:
                    pareto_front.append(pair)
            else:
                if pair[1] <= pareto_front[-1][1]:
                    pareto_front.append(pair)
        results[method].append(pareto_front)

    '''Plotting process'''
    plt.scatter(results['default'][1], results['default'][0], color='gold', label='Default CrossWalk')
    plt.scatter(results['ssa'][1], results['ssa'][0], color='skyblue', label='SSA CrossWalk')

    pf_X_default = [pair[0] for pair in results['default'][2]]
    pf_Y_default = [pair[1] for pair in results['default'][2]]

    pf_X_ssa = [pair[0] for pair in results['ssa'][2]]
    pf_Y_ssa = [pair[1] for pair in results['ssa'][2]]

    plt.plot(pf_X_default, pf_Y_default, color='r', label='Default CrossWalk')
    plt.plot(pf_X_ssa, pf_Y_ssa, color='g', label='SSA CrossWalk')

    plt.title('Pareto Front of CrossWalk vs SSA CrossWalk candidates\nNode Classification - Rice-Facebook dataset')
    plt.xlabel("Disparity")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f'{dataset}_pareto_front.png')


def main():
    all_files = [join("results", file) for file in listdir("results") if isfile(join("results", file))]
    for dataset in DATASETS:
        dataset_results_files = [file for file in all_files if dataset in file]
        unweighted_results_file = [file for file in dataset_results_files if "unweighted" in file][0]
        # fairwalk_results_file = [file for file in dataset_results_files if "fairwalk" in file][0]
        print(dataset_results_files)
        random_walk_results_files = [file for file in dataset_results_files if "subset_random_walk" in file]
        soft_results_file = [file for file in dataset_results_files if "_c0" in file]

        print(soft_results_file)
        print(random_walk_results_files)

        unweighted = read_results_txt(unweighted_results_file)
        # fairwalk = read_results_txt(fairwalk_results_file)

        for file_soft, file_rw in zip(soft_results_file, random_walk_results_files):
            output_filename = file_rw[ : -4] + "png"
            output_filename = output_filename.replace("results", "fig")
            print(file_soft)
            print(file_rw)
            print()
            rw_method = read_results_txt(file_rw)
            srw_method = read_results_txt(file_soft)

            get_bar_plot(unweighted, srw_method, rw_method, output_filename)


if __name__ == "__main__":
    plot_pareto_frontier('rice_subset', maxX=False, maxY=True)
    main()
