import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

def read_embeddings(emb_file):
    emb = dict()
    with open(emb_file, 'r') as fin:
        for i_l, line in enumerate(fin):
            s = line.split()
            if i_l == 0:
                dim = int(s[1])
                continue
            emb[int(s[0])] = [float(x) for x in s[1:]]
    return emb, dim

def read_sensitive_attr(sens_attr_file, emb):
    sens_attr = dict()
    with open(sens_attr_file, 'r') as fin:
        for line in fin:
            s = line.split()
            id = int(s[0])
            if id in emb:
                sens_attr[id] = int(s[1])
    return sens_attr

def plot_graph(embedding_filename, sensitive_attr_filename, res_file):
    emb, dim = read_embeddings(embedding_filename)
    sens_attr = read_sensitive_attr(sensitive_attr_filename, emb)
    # todo for random walk embeddings, the sizes are different. fix this
    print(len(emb))
    print(len(sens_attr))
    assert len(emb) == len(sens_attr)

    n = len(emb)
    X = np.zeros([n, dim])
    z = np.zeros([n])
    for i, id in enumerate(emb):
        X[i, :] = np.array(emb[id])
        z[i] = sens_attr[id]

    # Plot the obtained embeddings with TSNE
    X_emb = TSNE().fit_transform(X)
    X_red = X[z == 1, :]
    X_blue = X[z == 0, :]
    X_emb_red = X_emb[z == 1, :]
    X_emb_blue = X_emb[z == 0, :]
    plt.scatter(X_emb_red[:, 0], X_emb_red[:, 1], color='r', s=5)
    plt.scatter(X_emb_blue[:, 0], X_emb_blue[:, 1], color='b', s=5)
    plt.savefig(res_file, bbox_inches='tight')


if __name__ == '__main__':
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    # Rice subset, unweighted
    # plot_graph(embedding_filename='../data/rice_subset/rice_subset.embeddings_unweighted_d32',
    #            sensitive_attr_filename='../data/rice_subset/rice_subset.attr',
    #            res_file='../results/TSNE_rice_subset.unweighted.png')

    # Rice subset, CrossWalk with exp = 2.0
    plot_graph(embedding_filename='../data/rice_subset/rice_subset.embeddings_random_walk_5_bndry_0.5_exp_2.0_d32',
               sensitive_attr_filename='../data/rice_subset/rice_subset.attr',
               res_file='../results/TSNE_rice_subset.random_walk_5_bndry_0.5_exp_2.0.png')

    # Rice subset, CrossWalk with exp = 4.0
    plot_graph(embedding_filename='../data/rice_subset/rice_subset.embeddings_random_walk_5_bndry_0.5_exp_2.0_d32',
               sensitive_attr_filename='../data/rice_subset/rice_subset.attr',
               res_file='../results/TSNE_rice_subset.random_walk_5_bndry_0.5_exp_2.0.png')

    # Synthetic dataset, unweighted
    # plot_graph(embedding_filename='../data/synth2/synthetic_n500_Pred0.7_Phom0.025_Phet0.001.embeddings_unweighted_d32',
    #            sensitive_attr_filename='../data/synth2/synthetic_n500_Pred0.7_Phom0.025_Phet0.001.attr',
    #            res_file='../results/TSNE_synthetic_n500_Pred0.7_Phom0.025_Phet0.001.unweighted.png')
    #
    # # Synthetic dataset, CrossWalk with exp 0.4
    # plot_graph(embedding_filename='../data/synth2/synthetic_n500_Pred0.7_Phom0.025_Phet0.001.embeddings_random_walk_5_bndry_0.5_exp_2.0_d32_',
    #            sensitive_attr_filename='../data/synth2/synthetic_n500_Pred0.7_Phom0.025_Phet0.001.attr',
    #            res_file='../results/TSNE_synthetic_n500_Pred0.7_Phom0.025_Phet0.001.random_walk_5_bndry_0.5_exp_2.0.png')
    #
    # # Synthetic layered dataset, unweighted
    # plot_graph(embedding_filename='../data/synthetic_3layers/synthetic_3layers_n500_Pred0.7_Phom0.025_Phet0.001.embeddings_unweighted_d32',
    #            sensitive_attr_filename='../data/synthetic_3layers/synthetic_3layers_n500_Pred0.7_Phom0.025_Phet0.001.attr',
    #            res_file='../results/TSNE_synthetic_3layers_n500_Pred0.7_Phom0.025_Phet0.001.unweighted.png')
    #
    # # Synthetic layered dataset, CrossWalk with exp = 0.2
    # plot_graph(embedding_filename='../data/synthetic_3layers/synthetic_3layers_n500_Pred0.7_Phom0.025_Phet0.001.embeddings_random_walk_5_bndry_0.5_exp_2.0_d32',
    #            sensitive_attr_filename='../data/synthetic_3layers/synthetic_3layers_n500_Pred0.7_Phom0.025_Phet0.001.attr',
    #            res_file='../results/TSNE_synthetic_3layers_n500_Pred0.7_Phom0.025_Phet0.001.random_walk_5_bndry_0.5_exp_2.0.png')
    #
    # # Synthetic layered dataset, CrossWalk with exp = 0.4
    # plot_graph(embedding_filename='../data/synthetic_3layers/synthetic_3layers_n500_Pred0.7_Phom0.025_Phet0.001.embeddings_random_walk_5_bndry_0.5_exp_4.0_d32',
    #            sensitive_attr_filename='../data/synthetic_3layers/synthetic_3layers_n500_Pred0.7_Phom0.025_Phet0.001.attr',
    #            res_file='../results/TSNE_synthetic_3layers_n500_Pred0.7_Phom0.025_Phet0.001.random_walk_5_bndry_0.5_exp_4.0.png')
    #
    #


    # Unweighted embeddings - without CrossWalk
    # emb_file = '../data/rice_subset/rice_subset.embeddings_unweighted_d32' + str(run_i)
    # res_file = '../results/TSNE_rice_subset.unweighted.pdf'

    # CrossWalk - b)
    # emb_file = '../data/rice_subset/rice_subset.embeddings_random_walk_5_bndry_0.5_exp_2.0_d32' + str(run_i)
    # res_file = '../results/TSNE_rice_subset.random_walk_5_bndry_0.5_exp_2.0.pdf'

    # CrossWalk - c)
    # emb_file = '../data/rice_subset/rice_subset.embeddings_random_walk_5_bndry_0.5_exp_4.0_d32' + str(run_i)
    # res_file = '../results/TSNE_rice_subset.random_walk_5_bndry_0.5_exp_4.0.pdf'
    # sens_attr_file = '../data/rice_subset/rice_sensitive_attr.txt'

    # Unweighted embeddings - synthetic layered dataset a)
    # emb_file = '../data/synthetic/synthetic_n500_Pred0.7_Phom0.025_Phet0.001.embeddings_unweighted_d32' + str(run_i)
    # res_file = '../results/synthetic_n500_Pred0.7_Phom0.025_Phet0.001.unweighted.pdf'

    # CrossWalk - synthetic dataset - old bndry 0.4
    # emb_file = '../data/synthetic/synthetic_n500_Pred0.7_Phom0.025_Phet0.001.embeddings_random_walk_5_bndry_0.5_exp_2.0_d32' + str(run_i)
    # res_file = '../results/synthetic_n500_Pred0.7_Phom0.025_Phet0.001.random_walk_5_bndry_0.5_exp_2.0.pdf'
    # sens_attr_file = '../data/synthetic/synthetic_n500_Pred0.7_Phom0.025_Phet0.001.attr'

    # Unweighted embeddings - without CrossWalk - synthetic layered dataset
    # emb_file = '../data/synthetic_3layers/synthetic_3layers_n500_Pred0.7_Phom0.025_Phet0.001.embeddings_unweighted_d32' + str(run_i)
    # res_file = '../results/synthetic_3layers_n500_Pred0.7_Phom0.025_Phet0.001.unweighted.pdf'

    # CrossWalk - synthetic layered dataset - p=2.0
    # emb_file = '../data/synthetic_3layers/synthetic_3layers_n500_Pred0.7_Phom0.025_Phet0.001.embeddings_random_walk_5_bndry_0.5_exp_2.0_d32' + str(run_i)
    # res_file = '../results/synthetic_3layers_n500_Pred0.7_Phom0.025_Phet0.001.random_walk_5_bndry_0.5_exp_2.0.pdf'

    # CrossWalk - synthetic layered dataset - p=4.0
    # emb_file = '../data/synthetic_3layers/synthetic_3layers_n500_Pred0.7_Phom0.025_Phet0.001.embeddings_random_walk_5_bndry_0.5_exp_4.0_d32' + str(run_i)
    # res_file = '../results/synthetic_3layers_n500_Pred0.7_Phom0.025_Phet0.001.random_walk_5_bndry_0.5_exp_4.0.pdf'
    # sens_attr_file = '../data/synthetic_3layers/synthetic_3layers_n500_Pred0.7_Phom0.025_Phet0.01.attr'




