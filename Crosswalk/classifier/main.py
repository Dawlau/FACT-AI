from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import pairwise_distances, accuracy_score, confusion_matrix, classification_report
import numpy as np
from argparse import ArgumentParser
import warnings
from tqdm import tqdm
import os
import json

label_type = 'college' # 'college_2' # 'major' #
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

def classify(method, dataset, nfiles=5):
    res_total = []
    res_1_total = []
    res_0_total = []
    res_diff_total = []
    res_var_total = []

    runs = 200

    for iter in tqdm(range(runs)):
        run_i = 1 + np.mod(iter, int(nfiles))

        filename = os.path.join(ROOT_DIR, "datahub", dataset)
        emb_file = os.path.join(filename, f"{dataset}.embeddings_{method}_d32_{str(run_i)}")
        # emb_file = os.path.join(filename, f"rice_subset.embeddings_{method}_d32_{str(run_i)}")

        attr_filename = f"{dataset}_sensitive_attr.txt"
        label_filename = f"{dataset}_raw.txt"
        # for (_, _, files) in os.walk(filename, topdown=False):
        #     for file in files:
        #         if file.endswith(".attr"):
        #             attr_filename = file

        # assert attr_filename is not None, f"No attribute filename found for {dataset}"
        label_file = os.path.join(filename, label_filename)
        sens_attr_file = os.path.join(filename, attr_filename)

        emb, dim = read_embeddings(emb_file)
        labels = read_labels(label_file, emb)
        sens_attr = read_sensitive_attr(sens_attr_file, emb)

        assert len(labels) == len(emb) == len(sens_attr)

        n = len(emb)
        X = np.zeros([n, dim])
        y = np.zeros([n])
        z = np.zeros([n])
        for i, id_ in enumerate(emb):
            X[i, :] = np.array(emb[id_])
            y[i] = labels[id_]
            z[i] = sens_attr[id_]

        idx = np.arange(n)
        np.random.shuffle(idx)
        n_train = int(n // 2)

        # Shuffle the data
        X = X[idx, :]
        y = y[idx]
        z = z[idx]

        # luca: did modification
        # X_train = X  # old
        X_train = X[:n_train]  # new
        # luca: did modification
        # y_train = np.concatenate([y[:n_train], -1*np.ones([n-n_train])])  # old
        y_train = y[:n_train]  # new

        X_test = X[n_train:]
        y_test = y[n_train:]
        z_test = z[n_train:]

        # X_train = X[idx[:n_train], :]
        # X_test = X[idx[n_train:], :]
        # y_train = y[idx[:n_train]]
        # y_test = y[idx[n_train:]]
        # z_test = z[idx[n_train:]]

        g = np.mean(pairwise_distances(X))
        clf = LabelPropagation(gamma=g, max_iter=2000).fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        # print(classification_report(y_test, y_pred, target_names=['class 0', 'class 1'], digits=3))
        # unique, counts = np.unique(y_train, return_counts=True)
        # print(np.asarray((unique, counts)))
        # print()

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
    method = args.method
    nfiles = args.nfiles
    dataset = args.dataset

    results = classify(method, dataset, nfiles=nfiles)
    result = {'total_acc': results[0], 'acc_1': results[1],
              'acc_0': results[2], 'diff': results[3], 'disparity': results[4]}
    results_filename = os.path.join("results", f"{dataset}_{method}.json")
    with open(results_filename, "w") as w:
        json.dump(result, w)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--nfiles", default=5,
                      help="set int for how many versions of each emb")
    parser.add_argument("--warnings", default=True,
                      help="set true for receiving warnings")
    parser.add_argument("--method", help="Method used for embeddings")
    parser.add_argument("--dataset", help="Dataset used")

    args = parser.parse_args()

    # turn warnings off by default
    if args.warnings == False:
        warnings.filterwarnings("ignore")

    main(args)
