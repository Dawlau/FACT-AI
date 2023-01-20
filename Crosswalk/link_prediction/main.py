from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle
import os
import random

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
            links = pickle.load(f)
    
    # since the link files are not binary:
    else:
        with open(links_file, 'r') as f:
            lines = f.readlines()
            lines = [line.split() for line in lines]

    lines = np.array(lines)

    # convert str values to float
    #links = [[float(l[0]), float(l[1])] for l in links if float(l[0]) in emb.keys() and float(l[1]) in emb.keys()]
    
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
    return links

def extract_features(u,v):
    return (u-v)**2
    # return np.array([np.sqrt(np.sum((u-v)**2))])

def get_train_test_links(links_path, train_ratio):
    with open(links_path, 'r') as f:
        lines = f.readlines()
        lines = [line.split() for line in lines]
        # shuffle data
        random.shuffle(lines)
        split_idx = round(len(lines)*train_ratio)
        
        
        # for twitter dataset:
            # stripping necessary
            # only taking the first two node values from twitter.links
        train_set = np.char.strip(np.array(lines[:split_idx]), chars='[],').astype(float)
        test_set = np.char.strip(np.array(lines[split_idx:]), chars='[],').astype(float)
    return train_set[:,:2], test_set[:,:2]

# add labels corresponding to the nodes
def label_node_links(links_arr, link_attr):
    n_nodes = len(links_arr[0])
    
    for i in range(n_nodes):
        cur_nodes = links_arr[:,i]
        cur_node_labels = np.vectorize(link_attr.get)(cur_nodes)
        links_arr = np.concatenate((links_arr, cur_node_labels.reshape(-1, 1)), axis=1)
    if n_nodes == 2:
        eq_labels = (links_arr[:,2] == links_arr[:,3])
        links_arr = np.concatenate((links_arr, eq_labels.reshape(-1,1)), axis=1)
    return links_arr

# balance data, such that there is an equal number of each pair
def balance_links(links_arr, label_pairs):
    min_matches = 1e20
    balanced_arr = np.array([])
    for pair in label_pairs:
        matches = len(np.where((links_arr[:,2] == pair[0]) & (links_arr[:,3] == pair[1]))[0])
        if matches < min_matches:
            min_matches = matches
    
    for pair in label_pairs:
        balanced_arr = np.append(balanced_arr, links_arr[np.where((links_arr[:,2] == pair[0]) & (links_arr[:,3] == pair[1]))][:min_matches])

    balanced_arr = balanced_arr.reshape(-1, 5)
    return balanced_arr

def run_link_prediction(dataset, emb_type, bndry, exp, d, all_labels, n_iters=6):
    
    #all_labels = [0, 1]
    
    label_pairs = [(int(all_labels[i]), int(all_labels[j])) for i in range(len(all_labels)) for j in range(len(all_labels))]
    accuracy_keys = label_pairs + ['max_diff', 'var', 'total']
    accuracy = {k : [] for k in accuracy_keys}
    for iter in [str(k) for k in range(1,n_iters)]:

        print('iter: ', iter)

<<<<<<< Updated upstream
        # filename = 'sample_4000_connected_subset/sample_4000_connected_subset'
        filename = '../data/rice_subset/rice_subset'
        # emb_file = filename + '.embeddings_unweighted_d32_' + iter
        # emb_file = filename + '.embeddings_fairwalk_d32_' + iter
        # emb_file = filename + '.randomembedding_d32_' + iter
        emb_file = filename + '.pmodified__embeddings_fairwalk_d32'
        sens_attr_file = filename + '.attr'
        train_links_file = filename + '_' + iter + '_trainlinks'
        test_links_file = filename + '_' + iter + '_testlinks'
=======
        filename = f'data/{dataset}'

        emb_filepath = f'{filename}/{dataset}.embeddings_{emb_type}_bndry_{bndry}_exp_{exp}_d{d}.walks.{iter}'
        sens_attr_filepath = filename + '/' + dataset + '.attr'
        links_filepath = f'{filename}/{dataset}.links'
        
        emb, dim = read_embeddings(emb_filepath)
        sens_attr = read_sensitive_attr(sens_attr_filepath, emb)
>>>>>>> Stashed changes

        # load train and test links using path
        # 50-50 split
        train_links, test_links = get_train_test_links(links_filepath, 0.5)

        # add 3rd and 4th column with respective label
        # this is later used to verify whether labels are correct
        train_links = label_node_links(train_links, sens_attr)
        test_links = label_node_links(test_links, sens_attr)

        # balance the links, such that there is an equal amount of each label pair
        train_links = balance_links(train_links, label_pairs)
        test_links = balance_links(test_links, label_pairs)

        for key in label_pairs + ['total']:
            if key == 'total':
                label_pairs = [(all_labels[i],all_labels[j]) for i in range(len(all_labels)) for j in range(len(all_labels))]
            else:
                l1, l2 = key
                key = (l1, l2)
                label_pairs = [key]
                if l1 != l2:
                    label_pairs.append((l2,l1))
            
            # check for valid labels
            # first two elems are the node ids, 3rd and 4th elem are respective groups (label)
            filtered_train_links = train_links
            filtered_test_links = [l for l in test_links if (l[2], l[3]) in label_pairs]

            clf = LogisticRegression(solver='lbfgs')

            # x: node --> node (x1 --> x2)
            # y: 1 if labels equal, 0 else

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

        print(accuracy)
        print()

    print(accuracy)
    print()
    for k in accuracy_keys:
        print(str(k) + ' | accuracy:', np.mean(accuracy[k]), '| std: ' + str(np.std(accuracy[k])) + '')


if __name__ == '__main__':


    dataset = 'twitter' #'rice_subset', 'twitter'
    
    emb_type = 'random_walk_5'
    bndry = 0.5
    exp = 2.0
    d = 128 #128, 32, 64, 92
    all_labels = [0, 1] if 'rice_subset' else [0, 1, 2]
    n_iters = 6

    run_link_prediction(dataset, emb_type, bndry, exp, d, all_labels, n_iters)

    # TODO: 
    # fix naming for different embedding types