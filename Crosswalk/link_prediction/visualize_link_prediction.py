import os
import matplotlib.pyplot as plt
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

import pandas as pd
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(f'{ROOT_DIR}/link_prediction/link_prediction_results.csv', index_col=None)

red_ = '#fab3ac'
blue_ = '#29a5e3'
cyan_ = '#d2f0f7'
green_ = '#a3f77e'
gray_ = '#dbdbdb'
purple_ = '#BF55EC'
yellow_ = '#F7CA18'



def get_all_bar_plots(datasets, boundary_vals, exp_vals, df=df, save_path='fig'):
    label_size = 27
    font_size = 24
    image_size = (12, 8.5)
    bar_width = 0.5
    legend_size = 25
    y_lim = [0, 100]

    for dataset in datasets:
        for boundary_val in boundary_vals:
            for exp in exp_vals:
                # filter parameters
                results = df[df['dataset'] == dataset]
                results = results[results['boundary_val'] == boundary_val]
                results = results[results['exp'] == exp]

                # seperate results
                acc_proposed = results[results['embedding_type'] == 'random_walk'].to_dict('records')[0]
                acc_fairwalk = results[results['embedding_type'] == 'fairwalk'].to_dict('records')[0]
                acc_unweighted = results[results['embedding_type'] == 'unweighted'].to_dict('records')[0]

                fig_out_path = os.path.join(ROOT_DIR, 'link_prediction', save_path, f'{dataset}_bndry_{boundary_val}_exp_{exp}.png')
                get_bar_plot(acc_proposed, acc_fairwalk, acc_unweighted, bar_width, font_size, label_size, image_size, y_lim, legend_size, fig_out_path)


def get_bar_plot(acc_proposed, acc_fairwalk, acc_unweighted, bar_width, fontsize, labelsize, imagesize, ylim, legend_size, fig_out_path):

    #labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
    #labels = ['A-A', 'B-B', 'A-B', 'Total']

    xu = [2 - bar_width, 2]
    xf = [5.5 - bar_width, 5.5]
    xp = [9 - bar_width, 9]


    fig, ax = plt.subplots()

    ax.bar(xu[0], acc_unweighted['total'], bar_width, color = purple_, edgecolor='black', label='Total Accuracy')
    ax.bar(xf[0], acc_fairwalk['total'], bar_width, color = purple_, edgecolor='black')
    ax.bar(xp[0], acc_proposed['total'], bar_width, color = purple_, edgecolor='black')
    ax.bar(xu[1], acc_unweighted['var'], bar_width, color = yellow_, edgecolor='black', label='Disparity')
    ax.bar(xf[1], acc_fairwalk['var'], bar_width, color = yellow_, edgecolor='black')
    ax.bar(xp[1], acc_proposed['var'], bar_width, color = yellow_, edgecolor='black')
    plt.legend(loc='upper right', prop={'size': legend_size}) #'upper left')

    ax.set_ylim(ylim)
    plt.xticks([2, 5.5, 9], ['DeepWalk', 'FairWalk', 'CrossWalk'], fontsize=legend_size)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')

    plt.ylabel('Accuracy', fontsize = labelsize)
    plt.rcParams.update({'font.size': fontsize})
    plt.yticks(fontsize=labelsize)
    fig.set_size_inches(imagesize[0], imagesize[1])
    fig.savefig(fig_out_path, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    datasets = ['rice_subset', 'twitter']
    boundary_vals = [0.5]
    exp_vals = [2.0]
    get_all_bar_plots(datasets, boundary_vals, exp_vals)
