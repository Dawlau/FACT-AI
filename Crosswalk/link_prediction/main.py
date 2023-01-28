from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle
import os
import random
from tqdm import tqdm
import pandas as pd

from multiprocessing import Process, Pool

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

def read_links(links_file, emb, binary=False):
    if binary==True:
        with open(links_file, 'rb') as f:
            lines = pickle.load(f)
    
    # since the link files are not binary:
    else:
        with open(links_file, 'r') as f:
            lines = f.readlines()
            lines = [line.split() for line in lines]

    #lines = np.array(lines)

    # convert str values to float
    if binary==True:
        #links = [[float(l[0]), float(l[1])] for l in lines if float(l[0]) in emb.keys() and float(l[1]) in emb.keys()]
        return [l for l in lines if l[0] in emb.keys() and l[1] in emb.keys()]
    else:    
        # more robust method for above, although inefficient
        links = []
        n_nodes = len(lines[0])
        n_matches = 0
        for line in lines:
            for elem in line:
                if elem in emb.keys():
                    n_matches += 1
            if n_matches == n_nodes:
                links.append(line)
    return np.array(links)

def extract_features(u,v):
    return (u-v)**2
    # return np.array([np.sqrt(np.sum((u-v)**2))])


# run w synth2 after 
def run_link_prediction(dataset, emb_type, rwl, bndrytype, bndry, exp, d, n_iters=6, test_ratio=0.5):
    
    #all_labels = [0, 1]
    if dataset in ('rice_subset', 'synth2'):
        all_labels = [0, 1]
    elif dataset in ('twitter', 'synth3'):
        all_labels = [0, 1, 2]                

    label_pairs = [(int(all_labels[i]), int(all_labels[j])) for i in range(len(all_labels)) for j in range(len(all_labels))]
    accuracy_keys = label_pairs + ['max_diff', 'var', 'total']
    accuracy = {k : [] for k in accuracy_keys}
    for iter in tqdm([str(k) for k in range(1,n_iters)]):
        filename = f'data/{dataset}'

        if emb_type == 'unweighted':
            full_method = emb_type
            test_links_filepath = f'{filename}/{emb_type}/links/{emb_type}_{dataset}_d{d}-test-{test_ratio}_{iter}.links'
            train_links_filepath = f'{filename}/{emb_type}/links/{emb_type}_{dataset}_d{d}-train-{test_ratio}_{iter}.links'
        else:
            full_method=f'{emb_type}_{rwl}_{bndrytype}_{bndry}_exp_{exp}'
            test_links_filepath = f'{filename}/{emb_type}/links/{emb_type}_{rwl}_bndry_{bndry}_exp_{exp}-{dataset}-test-{test_ratio}_{iter}.links'
            train_links_filepath = f'{filename}/{emb_type}/links/{emb_type}_{rwl}_bndry_{bndry}_exp_{exp}-{dataset}-train-{1-test_ratio}_{iter}.links'
        
        emb_filepath = f'{filename}/{emb_type}/embeddings_{1-test_ratio}train{test_ratio}test/{dataset}.embeddings_{full_method}_d{d}_{iter}'
        #sens_attr_filepath = filename + '/' + dataset + '.attr'
        
        emb, dim = read_embeddings(emb_filepath)
        #sens_attr = read_sensitive_attr(sens_attr_filepath, emb)

        train_links = read_links(train_links_filepath, emb, binary=True)
        test_links = read_links(test_links_filepath, emb, binary=True)

        print('----', iter, dataset, emb_type, ' boundary_val = ', bndry, ' exp = ', exp, '----')
        
        for key in label_pairs + ['total']:
            if key == 'total':  
                valid_edge_pairs = [(all_labels[i],all_labels[j]) for i in range(len(all_labels)) for j in range(len(all_labels))]
            else:
                l1, l2 = key
                key = (l1, l2)
                valid_edge_pairs = [key]
                if l1 != l2:
                    valid_edge_pairs.append((l2,l1))
            
            # check for valid labels
            # first two elems are the node ids, 3rd and 4th elem are respective groups (label)
            filtered_train_links = train_links
            filtered_test_links = [l for l in test_links if (l[2], l[3]) in valid_edge_pairs]

            # make multinomial
            clf = LogisticRegression(solver='lbfgs', multi_class='multinomial')

            # extract_features() takes the squared difference of two elements
            x_train = np.array([extract_features(np.array(emb[l[0]]), np.array(emb[l[1]])) for l in filtered_train_links])
            y_train = np.array([l[4] for l in filtered_train_links])
            
            x_test = np.array([extract_features(np.array(emb[l[0]]), np.array(emb[l[1]])) for l in filtered_test_links])
            y_test = np.array([l[4] for l in filtered_test_links])

            clf.fit(x_train, y_train)

            y_pred = clf.predict(x_test)
            curr_acc = 100 * np.sum(y_test == y_pred) / x_test.shape[0]

            accuracy[key].append(curr_acc)
            if l1 != l2:
                accuracy[(l2, l1)].append(curr_acc)
    
        last_accs = [accuracy[k][-1] for k in label_pairs]
        accuracy['max_diff'].append(np.max(last_accs) - np.min(last_accs))
        accuracy['var'].append(np.var(last_accs))

    avg_accuracies = {}    
    print(f'embedding type: {emb_type} | rwl: {rwl} | bndry: {bndry} | bndrytype: {bndrytype} | exp: {exp} | d: {d}')
    for k in accuracy_keys:
        print(str(k) + ' | accuracy:', np.mean(accuracy[k]), '| std: ' + str(np.std(accuracy[k])) + '')
        avg_accuracies[k] = np.mean(accuracy[k])
    return (avg_accuracies, {'dataset':dataset, 'embedding_type':emb_type, 'rwl':rwl, 'boundary_type':bndrytype, 'boundary_val':bndry, 'exp':exp, 'd': d})

def experiment_parameters(datasets, emb_type, r, bndrytype, bndry, exp, d, threads=0, test_ratio=0.5, n_iters=6):
    if threads <= 1:
        accuracies = []
        for dataset in datasets:
            for emb_type in emb_types:
                for r in rwl:
                    for bndry in bndries:
                        for exp in exponents:
                            for bndrytype in bndry_types:
                                for d in ds:
                                    accuracy, args= run_link_prediction(dataset, emb_type, r, bndrytype, bndry, exp, d, n_iters=n_iters, test_ratio=test_ratio)
                                    accuracy['dataset'] = args['dataset']
                                    accuracy['embedding_type'] = args['embedding_type']
                                    accuracy['rwl'] = args['rwl']
                                    accuracy['boundary_val'] = args['boundary_val']
                                    accuracy['boundary_type'] = args['boundary_type']
                                    accuracy['exp'] = args['exp']
                                    accuracy['d'] = args['d']
                                    accuracies.append(accuracy)
    else:
        input_args = []
        for dataset in datasets:
            for emb_type in emb_types:
                for r in rwl:
                    for bndry in bndries:
                        for exp in exponents:
                            for bndrytype in bndry_types:
                                for d in ds:
                                    input_args.append((dataset, emb_type, r, bndrytype, bndry, exp, d, n_iters, test_ratio))

        accuracies = []
        with Pool(processes=threads) as pool:
            for result in pool.starmap(run_link_prediction, input_args):
                accuracy, args = result
                accuracy['dataset'] = args['dataset']
                accuracy['embedding_type'] = args['embedding_type']
                accuracy['rwl'] = args['rwl']
                accuracy['boundary_val'] = args['boundary_val']
                accuracy['boundary_type'] = args['boundary_type']
                accuracy['exp'] = args['exp']
                accuracy['d'] = args['d']
                accuracies.append(accuracy)
    return accuracies

if __name__ == '__main__':

    # TODO
    # now run with different parameters for:
    # bndries (alpha)
    # exponents (p)

    # done:
    # bndries (alpha) = 0.5
    # exponent (p) = 2.0
    # for twitter, rice, synth2/3

    datasets = ['twitter', 'rice_subset', 'synth2', 'synth3']
    rwl = [5] #[5, 10, 20]

    # bndries are referred to as alpha in the paper
    bndries = [0.5, 0.7, 0.9]

    # exponents are referred to as p in the paper
    exponents = [1.0, 2.0, 3.0, 4.0]
    bndry_types = ['bndry']
    emb_types = ['random_walk', 'fairwalk', 'unweighted']
    ds = [32] #128, 32, 64, 92
    
    # print single example
    #dataset = 'twitter' #'rice_subset', 'twitter'
    #accuracies = run_link_prediction(dataset, emb_types[0], rwl[0], bndry_types[0], bndries[0], exponents[0], ds[0], all_labels)
    #print(accuracies)
    
    test_ratio = 0.5
    accuracies = experiment_parameters(datasets, emb_types, rwl, bndry_types, bndries, exponents, ds, threads=0, test_ratio=test_ratio)
    df = pd.DataFrame.from_dict(accuracies)
    print(df)
    df.to_csv(f'link_prediction/link_prediction_results_test_ratio_{test_ratio}.csv', index=None)
    
    # p default 1
    # alpha = bndries