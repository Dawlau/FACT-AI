from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import pairwise_distances
import numpy as np
from argparse import ArgumentParser
import warnings

label_type = 'college' # 'college_2' # 'major' #

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

def read_labels(label_file, emb):
    labels = dict()
    with open(label_file, 'r') as fin:
        for line in fin:
            s = line.split()
            id = int(s[0])
            if id in emb:
                if label_type == 'major':
                    labels[id] = int(s[3])
                elif label_type == 'college':
                    labels[id] = int(s[1])
                elif label_type == 'college_2':
                    tmp = int(s[1])
                    if tmp > 5:
                        labels[id] = 1
                    else:
                        labels[id] = 0
                else:
                    raise Exception('unknown label_type')
    return labels


def read_sensitive_attr(sens_attr_file, emb):
    sens_attr = dict()
    with open(sens_attr_file, 'r') as fin:
        for line in fin:
            s = line.split()
            id = int(s[0])
            if id in emb:
                sens_attr[id] = int(s[1])
    return sens_attr

def classify(method, ablation=False, nfiles=5):
    res_total = []
    res_1_total = []
    res_0_total = []
    res_diff_total = []
    res_var_total = []

    runs = 10

    for iter in range(runs):
        run_i = 1 + np.mod(iter, int(nfiles))

        if ablation:
            emb_file = '../data/rice_subset/ablation/rice_subset.embeddings_' + method + '_' + str(run_i)
        else:
            emb_file = '../data/rice_subset/rice_subset.embeddings_' + method + '_' + str(run_i)
        label_file = '../data/rice_subset/rice_subset.attr'
        sens_attr_file = '../data/rice_subset/rice_subset.attr'

        emb, dim = read_embeddings(emb_file)
        labels = read_labels(label_file, emb)
        sens_attr = read_sensitive_attr(sens_attr_file, emb)

        assert len(labels) == len(emb) == len(sens_attr)

        n = len(emb)

        X = np.zeros([n, dim])
        y = np.zeros([n])
        z = np.zeros([n])
        for i, id in enumerate(emb):
            X[i,:] = np.array(emb[id])
            y[i] = labels[id]
            z[i] = sens_attr[id]

        idx = np.arange(n)
        np.random.shuffle(idx)
        n_train = int(n // 2)
        
        X = X[idx,:]
        y = y[idx]
        z = z[idx]
        X_train = X
        X_test = X[n_train:]
        y_train = np.concatenate([y[:n_train], -1*np.ones([n-n_train])])
        y_test = y[n_train:]
        z_test = z[n_train:]
        g = np.mean(pairwise_distances(X))
        clf = LabelPropagation(gamma = g, max_iter=2000).fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        

        res = 100 * np.sum(y_pred == y_test) / y_test.shape[0]

        idx_1 = (z_test == 1)
        res_1 = 100 * np.sum(y_pred[idx_1] == y_test[idx_1]) / np.sum(idx_1)

        idx_0 = (z_test == 0)
        res_0 = 100 * np.sum(y_pred[idx_0] == y_test[idx_0]) / np.sum(idx_0)

        res_diff = np.abs(res_1 - res_0)
        res_var = np.var([res_1, res_0])

        res_total.append(res)
        res_1_total.append(res_1)
        res_0_total.append(res_0)
        res_diff_total.append(res_diff)
        res_var_total.append(res_var)

    res_avg = np.mean(np.array(res_total), axis=0)
    res_1_avg = np.mean(np.array(res_1_total), axis=0)
    res_0_avg = np.mean(np.array(res_0_total), axis=0)
    res_diff_avg = np.mean(np.array(res_diff_total), axis=0)
    res_var_avg = np.mean(np.array(res_var_total), axis=0)

    return res_avg, res_1_avg, res_0_avg, res_diff_avg, res_var_avg

def main(args):
    if args.ablation:
        results = {}
        # classify for a = [0.0 ...] in case of ablation study for method crosswalk
        for a in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            method = 'random_walk_5_bndry_' + str(a) + '_exp_2.0'
            res = classify(method, ablation=True, nfiles=args.nfiles)
            print('ablation report for a ' + method + ': ', res)
            results[method] = np.array(res)
        # save results to file in dir
        np.save('../data/rice_subset/ablation/results.npy', results) 
    else:
        results = {}
        # classify for all methods (no ablation study of 'a')
        for method in ['random_walk_5_bndry_0.5_exp_2.0', 'unweighted', 'fairwalk']:
            res = classify(method)
            print('classification report for method ' + method + ': ', res)
            results[method] = np.array(res)
        # save results to file in dir
        np.save('../data/rice_subset/results.npy', results) 


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--ablation", default=False,
                      help="set true for ablation study")
    parser.add_argument("--nfiles", default=5,
                      help="set int for how many versions of each emb")
    parser.add_argument("--warnings", default=False,
                      help="set true for receiving warnings")
    args = parser.parse_args()

    # turn warnings off by default
    if args.warnings == False:
        warnings.filterwarnings("ignore")

    main(args)
    
    