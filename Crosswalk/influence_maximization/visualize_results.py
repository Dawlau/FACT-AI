import os
import matplotlib.pyplot as plt
import numpy as np
import json
from os import listdir
from os.path import isfile, join
from collections import defaultdict
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


DATASETS = ["soft_rice_subset", "soft_synth2", "soft_synth3", "rice_subset", "synth2", "synth3"]
METHODS = ["adv", "unweighted", "fairwalk", "random_walk", "greedy"]
NUM_NODES_A = {"soft_rice_subset": 97, "soft_synth2": 150, "soft_synth3": 125,
               "rice_subset": 97, "synth2": 150, "synth3": 125, "twitter": 2598}
NUM_NODES_B = {"soft_rice_subset": 344, "soft_synth2": 350, "soft_synth3": 300,
               "rice_subset": 344, "synth2": 350, "synth3": 300, "twitter": 78}
NUM_NODES_C = {"synth3": 75, "twitter": 180,
               "soft_synth2": 75}
NUM_GROUPS = {"soft_rice_subset": 2, "soft_synth2": 2, "soft_synth3": 3,
              "rice_subset": 2, "synth2": 2, "synth3": 3, "twitter": 3}

n_seeds = np.arange(2, 41, 2)
red_ = '#fab3ac'
blue_ = '#29a5e3'
cyan_ = '#d2f0f7'
green_ = '#a3f77e'
gray_ = '#dbdbdb'
purple_ = '#BF55EC'
yellow_ = '#F7CA18'
label_size = 27
font_size = 24
image_size = (14, 8.5)
line_width = 3
square = True
bar_width = 0.5
legend_size = 20


def get_bar_plot_with_greedy(total_influence_results, disparity_results, dataset, walk, ylim=None):
    has_3_groups = NUM_GROUPS[dataset] == 3

    xe = [2 - bar_width, 2]
    xu = [4 - bar_width, 4]
    xf = [6 - bar_width, 6]
    xa = [8 - bar_width, 8]

    if not has_3_groups:
        xp = [10 - bar_width, 10]

    deepwalk_influence = total_influence_results["unweighted"]
    fairwalk_influence = total_influence_results["fairwalk"]
    crosswalk_influence = total_influence_results[walk]
    greedy_influence = total_influence_results["greedy"]

    if not has_3_groups:
        adv_influence = total_influence_results["adv"]

    deepwalk_disparity = disparity_results["unweighted"]
    fairwalk_disparity = disparity_results["fairwalk"]
    crosswalk_disparity = disparity_results[walk]
    greedy_disparity = disparity_results["greedy"]

    if not has_3_groups:
        adv_disparity = disparity_results["adv"]

    fig, ax = plt.subplots()

    # ax.bar(xe[0], greedy_influence, bar_width, color=purple_, edgecolor='black', label='Total Influence Percentage')
    ax.bar(xu[0], deepwalk_influence, bar_width, color=purple_, edgecolor='black')
    ax.bar(xf[0], fairwalk_influence, bar_width, color=purple_, edgecolor='black')
    ax.bar(xa[0], crosswalk_influence, bar_width, color=purple_, edgecolor='black')
    if not has_3_groups:
        ax.bar(xp[0], adv_influence, bar_width, color=purple_, edgecolor='black')

    # ax.bar(xe[1], greedy_disparity, bar_width, color=yellow_, edgecolor='black', label='Disparity')
    ax.bar(xu[1], deepwalk_disparity, bar_width, color=yellow_, edgecolor='black')
    ax.bar(xf[1], fairwalk_disparity, bar_width, color=yellow_, edgecolor='black')
    ax.bar(xa[1], crosswalk_disparity, bar_width, color=yellow_, edgecolor='black')
    if not has_3_groups:
        ax.bar(xp[1], adv_disparity, bar_width, color=yellow_, edgecolor='black')

    if ylim:
        ax.set_ylim([0, 100])

    plt.legend(loc='upper right', prop={'size': legend_size})

    if has_3_groups:
        plt.xticks([2, 4, 6, 8], ['Greedy', 'DeepWalk', 'FairWalk', 'CrossWalk'], fontsize=legend_size)
    else:
        plt.xticks([2, 4, 6, 8, 10], ['Greedy', 'DeepWalk', 'FairWalk', 'CrossWalk', 'Adversarial'],
                   fontsize=legend_size)

    ax.yaxis.grid(color='gray', linestyle='dashed')

    plt.rcParams.update({'font.size': font_size})
    plt.yticks(fontsize=label_size)
    fig.set_size_inches(image_size[0], image_size[1])

    fig.savefig(os.path.join("fig", dataset) + f"_{walk}_greedy.pdf", bbox_inches='tight')
    plt.close()


def get_bar_plot_without_greedy(total_influence_results, disparity_results, dataset, walk, ylim=None):
    has_3_groups = NUM_GROUPS[dataset] == 3

    xe = [2 - bar_width, 2]
    xu = [4 - bar_width, 4]
    xf = [6 - bar_width, 6]

    if not has_3_groups:
        xp = [8 - bar_width, 8]

    deepwalk_influence = total_influence_results["unweighted"]
    fairwalk_influence = total_influence_results["fairwalk"]
    crosswalk_influence = total_influence_results[walk]

    if not has_3_groups:
        adv_influence = total_influence_results["adv"]

    deepwalk_disparity = disparity_results["unweighted"]
    fairwalk_disparity = disparity_results["fairwalk"]
    crosswalk_disparity = disparity_results[walk]

    if not has_3_groups:
        adv_disparity = disparity_results["adv"]

    fig, ax = plt.subplots()

    ax.bar(xe[0], deepwalk_influence, bar_width, color=purple_, edgecolor='black', label='Total Influence Percentage')
    ax.bar(xu[0], fairwalk_influence, bar_width, color=purple_, edgecolor='black')
    ax.bar(xf[0], crosswalk_influence, bar_width, color=purple_, edgecolor='black')
    if not has_3_groups:
        ax.bar(xp[0], adv_influence, bar_width, color=purple_, edgecolor='black')

    ax.bar(xe[1], deepwalk_disparity, bar_width, color=yellow_, edgecolor='black', label='Disparity')
    ax.bar(xu[1], fairwalk_disparity, bar_width, color=yellow_, edgecolor='black')
    ax.bar(xf[1], crosswalk_disparity, bar_width, color=yellow_, edgecolor='black')
    if not has_3_groups:
        ax.bar(xp[1], adv_disparity, bar_width, color=yellow_, edgecolor='black')

    if ylim:
        ax.set_ylim([0, ylim])

    plt.legend(loc='upper right', prop={'size': legend_size})

    if has_3_groups:
        plt.xticks([2, 4, 6], ['DeepWalk', 'FairWalk', 'CrossWalk'], fontsize=legend_size)
    else:
        plt.xticks([2, 4, 6, 8], ['DeepWalk', 'FairWalk', 'CrossWalk', 'Adversarial'], fontsize=legend_size)

    ax.yaxis.grid(color='gray', linestyle='dashed')

    plt.rcParams.update({'font.size': font_size})
    plt.yticks(fontsize=label_size)
    fig.set_size_inches(image_size[0], image_size[1])

    fig.savefig(os.path.join("fig", dataset) + f"_{walk}_no_greedy.pdf", bbox_inches='tight')
    plt.close()


def read_txt_file(filename, dataset):
    has_3_groups = NUM_GROUPS[dataset] == 3

    n_a = NUM_NODES_A[dataset]
    n_b = NUM_NODES_B[dataset]

    if has_3_groups:
        n_c = NUM_NODES_C[dataset]

    inf_a, inf_b, inf_c = [], [], []

    with open(filename, "r") as r:
        for line in r:
            info = line.split()

            inf_a.append(float(info[2]))
            inf_b.append(float(info[4]))

            # print('inf_a.append(float(info[2]))', float(info[2]))
            # print('inf_b.append(float(info[4]))', float(info[4]))

            if has_3_groups:
                inf_c.append(float(info[6]))
                # print('inf_c.append(float(info[6]))', float(info[6]))

    inf_a, inf_b = np.array(inf_a), np.array(inf_b)
    if has_3_groups:
        inf_c = np.array(inf_c)

    if has_3_groups:
        total_fraction = 100 * (inf_a + inf_b + inf_c) / (n_a + n_b + n_c)
    else:
        total_fraction = 100 * (inf_a + inf_b) / (n_a + n_b)

    frac_a = 100 * inf_a / n_a
    frac_b = 100 * inf_b / n_b

    if has_3_groups:
        frac_c = 100 * inf_c / n_c

    if has_3_groups:
        var_fraction = np.var(np.concatenate([(100 * inf_a / n_a).reshape([-1, 1]),
                                              (100 * inf_b / n_b).reshape([-1, 1]),
                                              (100 * inf_c / n_c).reshape([-1, 1])],
                                             axis=1), axis=1)
    else:
        var_fraction = np.var(np.concatenate([(100 * inf_a / n_a).reshape([-1, 1]),
                                              (100 * inf_b / n_b).reshape([-1, 1])],
                                             axis=1), axis=1)

    frac_a = 100 * inf_a / n_a
    frac_b = 100 * inf_b / n_b

    results = np.concatenate([np.array(total_fraction).reshape([-1, 1]),
                              # np.array(frac_a).reshape([-1, 1]),
                              # np.array(frac_b).reshape([-1, 1]),
                              np.array(var_fraction).reshape([-1, 1])],
                             axis=1)

    return results


def plot_pareto_frontier(dataset, maxX=True, maxY=True):
    data_paths = [data_path for data_path in listdir("results") if dataset in data_path]
    results = defaultdict(lambda: [[], []])

    for data_path in data_paths:
        all_files = [join(f"results/{data_path}", file)
                     for file in listdir(f"results/{data_path}")
                     if isfile(join(f"results/{data_path}", file))
                        and 'random_walk' in file]
        for file in all_files:
            inf, disp = read_txt_file(file, dataset)[-1]
            if 'soft' in file:
                results['ssa'][0].append(inf)
                results['ssa'][1].append(disp)
            else:
                results['default'][0].append(inf)
                results['default'][1].append(disp)

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

    plt.title(f'Pareto Front of CrossWalk vs SSA CrossWalk candidates\nInfluence Maximization - {dataset} dataset')
    plt.xlabel("Disparity")
    plt.ylabel("Total Influence Percentage")
    plt.legend()
    plt.savefig(f'{dataset}_pareto_front.png')
    plt.show()


def main(without_greedy):
    for dataset in DATASETS:
        result_files = os.listdir(os.path.join("results", dataset))
        all_random_walks = {file.replace("_results.txt", "")[: -2] for file in result_files if "random_walk" in file}

        total_influence_results = {}
        disparity_results = {}

        print(result_files)

        for method in METHODS:
            if method != "adv":
                if method != "random_walk":
                    cur_files = [os.path.join("results", dataset, file) for file in result_files if method in file]

                    all_results = []
                    for file in cur_files:
                        results = read_txt_file(file, dataset)
                        all_results.append(results)

                    all_results = np.mean(np.concatenate([np.expand_dims(result, 2) for result in all_results], axis=2),
                                          axis=2)
                    total_influence_results[method] = all_results[-1][0]
                    disparity_results[method] = all_results[-1][1]
                else:
                    for walk in all_random_walks:
                        cur_files = [os.path.join("results", dataset, file) for file in result_files if walk in file]
                        all_results = []
                        print('cur_files', cur_files)
                        for file in cur_files:
                            results = read_txt_file(file, dataset)
                            all_results.append(results)

                        all_results = np.mean(
                            np.concatenate([np.expand_dims(result, 2) for result in all_results], axis=2), axis=2)
                        total_influence_results[walk] = all_results[-1][0]
                        disparity_results[walk] = all_results[-1][1]
            elif NUM_GROUPS[dataset] == 2:
                results_filename = os.path.join("results", dataset, "adv_results.txt")

                with open(results_filename, "r") as r:
                    results = json.load(r)

                    influence_results = []
                    disparities = []

                    for result in results:
                        influence_a = result[4] * 100
                        influence_b = result[5] * 100
                        total_influence = (influence_a + influence_b) / 2
                        disparity = np.var([influence_a, influence_b])

                        influence_results.append(total_influence)
                        disparities.append(disparity)

                    total_influence = np.mean(influence_results)
                    disparity = np.mean(disparities)
                    print(disparity)

                disparity_results[method] = disparity
                total_influence_results[method] = total_influence

        for walk in disparity_results.keys():
            if "random_walk" in walk:
                if without_greedy:
                    get_bar_plot_without_greedy(total_influence_results, disparity_results, dataset, walk)
                else:
                    print(walk)
                    get_bar_plot_with_greedy(total_influence_results, disparity_results, dataset, walk)


if __name__ == "__main__":
    parser = ArgumentParser("Visualize results for influence maximization",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')

    parser.add_argument('--without_greedy', action="store_true",
                        help='Flag that specifies whether to use greedy or not', default=False)
    args = parser.parse_args()

    # plot_pareto_frontier('synth2', maxX=False, maxY=True)
    # plot_pareto_frontier('synth3', maxX=False, maxY=True)
    # plot_pareto_frontier('rice_subset', maxX=False, maxY=True)
    main(args.without_greedy)
