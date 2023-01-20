import numpy as np
import networkx as nx
import pandas as pd
import tensorflow as tf

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Embedding
from tensorflow.python.ops.losses import losses

from sklearn.cluster import KMeans
from tensorflow.python.client import device_lib
import time


def build_encoder(embedding_size):
  model = Sequential()

  # The first encoder layer
  model.add(Dense(embedding_size*4, activation='relu'))

  # The second encoder layer
  model.add(Dense(embedding_size*2, activation='relu'))

  # The output layer
  model.add(Dense(embedding_size, activation='relu'))

  return model


def build_decoder(embedding_size, output_size):
  model = Sequential()

  # The first decoder layer
  model.add(Dense(embedding_size*2, activation='relu'))

  # The secod decoder layer
  model.add(Dense(embedding_size*4, activation='relu'))

  # The third decoder layer
  model.add(Dense(output_size, activation='sigmoid'))

  return model


def build_ae(encoder, decoder, output_size):

  input_tensor = Input(shape=(output_size,))
  embeddings = encoder(input_tensor)
  reconstructions = decoder(embeddings)

  auto_encoder = Model(input_tensor, reconstructions)

  return auto_encoder


def recon_loss(x, x_hat):

  return tf.reduce_sum(tf.keras.losses.binary_crossentropy(x, x_hat))


def first_order_loss(X, Z):

    X = tf.cast(X, tf.float32)
    Z= tf.cast(Z, tf.float32)

    D = tf.linalg.diag(tf.reduce_sum(X,1))
    L = D - X ## L is laplation-matriX

    return 2*tf.linalg.trace(tf.matmul(tf.matmul(tf.transpose(Z),L),Z))


def ae_adversarial_loss(X, Z, x, x_hat, d_z0, d_z1, first_order, alpha):

  # Recon loss
  reccon_loss = recon_loss(x, x_hat)
  f1_loss = first_order_loss(X, Z)

  if first_order =='with_f1':
      reccon_loss += alpha * f1_loss

  ### Loss 2 -> Same as the loss of the generator
  adversarial_loss = tf.reduce_sum(tf.keras.losses.binary_crossentropy(tf.ones_like(d_z0), d_z0)) + \
                     tf.reduce_sum(tf.keras.losses.binary_crossentropy(tf.zeros_like(d_z1), d_z1))

  return  reccon_loss  + 10 * adversarial_loss


def ae_accuracy(x, x_hat):
  round_x_hat = tf.round(x_hat)
  return tf.reduce_mean(tf.cast(tf.equal(x, round_x_hat), tf.float32))


def pretrain_step_embd(X, x, encoder, decoder, auto_encoder, pre_optimizer, first_order, alpha):

  with tf.GradientTape() as pre_tape:
    z = encoder(x, training=True)
    x_hat = decoder(z, training=True)

    Z = encoder(X, training=True)

    pre_loss = recon_loss(x, x_hat)

    if first_order == 'with_f1':
        pre_loss += alpha * first_order_loss(X, Z)


  pre_gradients = pre_tape.gradient(pre_loss, auto_encoder.trainable_variables)
  pre_optimizer.apply_gradients(zip(pre_gradients, auto_encoder.trainable_variables))

  pre_acc = ae_accuracy(x, x_hat)

  return tf.reduce_mean(pre_loss), pre_acc


def pretrain_embd(X, idxs, encoder, decoder, auto_encoder, pre_optimizer, first_order, alpha):
  np.random.shuffle(idxs)
  PRE_EPOCHS = 300
  Batch_size = 50

  for epoch in range(PRE_EPOCHS):

    epoch_losses = []
    epoch_acc = []

    for batch_idx in range(0, len(idxs), Batch_size):

      selected_idxs = idxs[batch_idx : batch_idx + Batch_size]
      adjacency_batch = X[selected_idxs, :]

      loss, accuracy= pretrain_step_embd(X, tf.cast(adjacency_batch, tf.float32), encoder, decoder, auto_encoder, pre_optimizer, first_order, alpha)

      epoch_losses.append(loss)
      epoch_acc.append(accuracy)


def build_discriminator(embedding_size):
  model = Sequential()

  # The input layer
  model.add(Input(shape=(embedding_size,)))

  # The first hidden layer
  model.add(Dense(25, activation='relu'))
  model.add(Dropout(0.25))

  # The second layer
  model.add(Dense(15, activation='relu'))
  model.add(Dropout(0.20))

  # The third layer
  model.add(Dense(6, activation='relu'))
  model.add(Dropout(0.20))

  model.add(Dense(1, activation = 'sigmoid'))

  return model


def disc_loss_function(d_z0, d_z1):

  loss_zero = tf.keras.losses.binary_crossentropy(tf.zeros_like(d_z0), d_z0)
  loss_one = tf.keras.losses.binary_crossentropy(tf.ones_like(d_z1), d_z1)

  return tf.cast(loss_zero, tf.float32) + tf.cast(loss_one, tf.float32)


def train_step(X, x0, x1, encoder, decoder, auto_encoder, discriminator, ae_optimizer, disc_optimizer, first_order, alpha):
  with tf.GradientTape() as ae_tape, tf.GradientTape() as disc_tape:

    z0 = encoder(x0, training=True)
    z1 = encoder(x1, training=True)

    Z = encoder(X, training=True)

    d_z0 = discriminator(z0, training=True)
    d_z1 = discriminator(z1, training=True)

    x0_hat = decoder(z0, training=True)
    x1_hat = decoder(z1, training = True)


    ae_loss = ae_adversarial_loss(X, Z, tf.concat([x0, x1], 0), tf.concat([x0_hat, x1_hat], 0), d_z0, d_z1, first_order, alpha)
    disc_loss = disc_loss_function(d_z0, d_z1)


  gradients_ae = ae_tape.gradient(ae_loss, auto_encoder.trainable_variables)
  gradients_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

  ae_optimizer.apply_gradients(zip(gradients_ae, auto_encoder.trainable_variables))
  disc_optimizer.apply_gradients(zip(gradients_disc, discriminator.trainable_variables))

  ae_acc = ae_accuracy(tf.concat([x0, x1], 0), tf.concat([x0_hat, x1_hat], 0))

  return tf.reduce_mean(ae_loss), ae_acc, tf.reduce_mean(disc_loss)


def pretrain_step_disc(x0, x1, encoder, discriminator, disc_pre_optimizer):

  z0 = encoder(x0)
  z1 = encoder(x1)

  with tf.GradientTape() as disc_tape_sep:

    d_z0= discriminator(z0, training=True)
    d_z1 = discriminator(z1, training=True)

    disc_loss = disc_loss_function(d_z0, d_z1)


  gradients_disc = disc_tape_sep.gradient(disc_loss, discriminator.trainable_variables)
  disc_pre_optimizer.apply_gradients(zip(gradients_disc, discriminator.trainable_variables))

  return tf.reduce_mean(disc_loss)


def pretrain_disc(X, idxs_zeros, idxs_ones, encoder, discriminator, disc_pre_optimizer):

  EPOCHS = 40

  np.random.shuffle(idxs_zeros)
  np.random.shuffle(idxs_ones)
  Batch_size = 50

  for epoch in range(EPOCHS):
    for batch_idx in range(0, len(idxs_ones), Batch_size):

      selected_zeros = idxs_zeros[batch_idx : batch_idx + Batch_size]
      selected_ones = idxs_ones[batch_idx : batch_idx + Batch_size]

      x0 = X[selected_zeros]
      x1 = X[selected_ones]

      pretrain_step_disc(x0, x1, encoder, discriminator, disc_pre_optimizer)


def adversarial_train(X, idxs_zeros, idxs_ones, encoder, decoder, auto_encoder, discriminator, ae_optimizer, disc_optimizer, first_order, alpha):
    EPOCHS = 500

    np.random.shuffle(idxs_zeros)
    np.random.shuffle(idxs_ones)

    Batch_size = 50

    for epoch in range(EPOCHS):
        for batch_idx in range(0, len(idxs_ones), Batch_size):

          selected_zeros = idxs_zeros[batch_idx : batch_idx + Batch_size]
          selected_ones = idxs_ones[batch_idx : batch_idx + Batch_size]

          x0 = X[selected_zeros]
          x1 = X[selected_ones]

          ### Joint Training
          train_step(X, tf.cast(x0, tf.float32), tf.cast(x1, tf.float32), encoder, decoder,
                                                     auto_encoder, discriminator, ae_optimizer, disc_optimizer, first_order, alpha)


def get_seeds(N_CLUS, embedding, nodes, labels, nodes_zero, nodes_one, strategy, n_seeds):
    '''
    stratgey can be random, nearest, fair, re-cluster, fair_re-cluster
    '''

    model = KMeans(n_clusters=N_CLUS)
    model.fit(embedding)

    cluster_number = model.labels_
    centers = model.cluster_centers_

    seed_ids = [[] for i in range(N_CLUS)]

    for i in range(N_CLUS):

        if strategy == 'nearest':
          sorted_distance = np.array(sorted([[np.sqrt(np.sum(np.power(centers[i] - embedding[j], 2))), j] for j in range(len(embedding)) if i == cluster_number[j]]))
          seed_ids[i].extend(list(sorted_distance[:n_seeds, 1]))


        elif strategy == 're-cluster':
          temp = []
          sorted_distance = np.array(sorted([[np.sqrt(np.sum(np.power(centers[i] - embedding[j], 2))), j] for j in range(len(embedding)) if i == cluster_number[j]]))
          temp.extend(list(sorted_distance[:n_seeds, 1]))

          portion_zero = 0
          portion_one = 0

          for num in temp:
            if num in nodes_zero:
              portion_zero += 1
            elif num in nodes_one:
              portion_one += 1

          zero_in_clus = embedding[np.logical_and(cluster_number == i, labels == 0)]
          zero_inds = nodes[np.logical_and(cluster_number == i, labels == 0)]

          one_in_clus = embedding[np.logical_and(cluster_number == i, labels == 1)]
          one_inds = nodes[np.logical_and(cluster_number == i, labels == 1)]

          added_to_zero = 0
          if len(zero_in_clus) != 0:
              model_on_zero = KMeans(n_clusters=1)
              model_on_zero.fit(zero_in_clus)
              center_zero = model_on_zero.cluster_centers_

              sorted_distance_zero = np.array(sorted([[np.sqrt(np.sum(np.power(center_zero - zero_in_clus[j], 2))), j] for j in range(len(zero_in_clus))]))
              seed_ids[i].extend([zero_inds[int(i)] for i in sorted_distance_zero[:portion_zero, 1]])

              added_to_zero = len(seed_ids[i])
              assert added_to_zero == portion_zero

          added_to_one = 0
          if len(one_in_clus) != 0:
              model_on_one = KMeans(n_clusters=1)
              model_on_one.fit(one_in_clus)
              center_one = model_on_one.cluster_centers_

              sorted_distance_one = np.array(sorted([[np.sqrt(np.sum(np.power(center_one - one_in_clus[j], 2))), j] for j in range(len(one_in_clus))]))
              seed_ids[i].extend([one_inds[int(i)] for i in sorted_distance_one[:portion_one, 1]])

              added_to_one = len(seed_ids[i]) - added_to_zero
              assert added_to_one == portion_one

          assert n_seeds == added_to_zero + added_to_one
          assert len(seed_ids[i]) == n_seeds

    return np.reshape(seed_ids, newshape=(-1, ))


def get_graph_real(edges_filename):
  graph_df = pd.read_csv(edges_filename, sep=" ", header=None)
  graph_df.columns = ['s', 't']

  edges = []

  for index, row in graph_df.iterrows():
    edge_cur = (row.s, row.t)
    edges.append(edge_cur)

  input_G = nx.from_edgelist(edges)

  unfrozen_G = nx.Graph(input_G)

  X = nx.to_numpy_matrix(unfrozen_G)
  G = nx.from_numpy_matrix(X)

  return G, X, unfrozen_G


def get_nodes_labels_real(input_G, attr_filename):
  data = pd.read_csv(attr_filename, sep=" ", header=None)
  data.columns = ['id', 'group']

  dict_labels = {}

  for index, row in data.iterrows():
    dict_labels[row.id] = row.group

  labels = []
  for node in list(input_G.nodes()):
    labels.append(dict_labels[node])

  # Remember that whenever you want do logical operation on a sequence, that sequence should be numpy array
  labels = np.array(labels)

  nodes = np.arange(len(input_G.nodes()))

  nodes_zero = nodes[labels == 0]
  nodes_one = nodes[labels == 1]

  return nodes_zero, nodes_one, labels


def get_idxs(n, nodes_zero, nodes_one):

  diff_size = np.abs(len(nodes_one) - len(nodes_zero))

  idxs_zeros = nodes_zero[:]
  idxs_ones = nodes_one[:]
  rep_policy = False

  if len(nodes_zero) < len(nodes_one):
    if diff_size > len(nodes_zero):
      rep_policy = True
    zero_draws = np.random.choice(nodes_zero,size=diff_size, replace=rep_policy)
    idxs_zeros = np.concatenate((idxs_zeros, zero_draws))

  elif len(nodes_zero) > len(nodes_one):
    if diff_size > len(nodes_one):
      rep_policy = True
    one_draws = np.random.choice(nodes_one,size=diff_size, replace=rep_policy)
    idxs_ones = np.concatenate((idxs_ones, one_draws))

  assert len(idxs_zeros) == len(idxs_ones)

  return np.arange(n), idxs_zeros, idxs_ones


def IC(G, seeds, imp_prob, recover_prob = 0, remove = 0):

    impressed = []
    removed = []
    front = list(seeds[:])

    while front:
        impressed.extend(front)
        impressed = np.array(impressed)

        if recover_prob != 0:

            random_draws = np.random.uniform(size=len(impressed))

            if remove:
                removed.extend(impressed[random_draws < recover_prob])
                removed = list(set(removed))

            impressed = impressed[random_draws >= recover_prob]

        impressed = list(impressed)
        new_front = []

        for node in front:

            neighbours = list(G.neighbors(node))

            for neigh in neighbours:

                expr_prob = np.random.uniform(size=1)[0]
                if expr_prob < imp_prob and not (neigh in impressed) and not (neigh in new_front) and not (neigh in removed):
                    new_front.append(neigh)

        front = new_front[:]

    impressed = np.reshape(np.array(impressed), newshape=(-1,))

    return impressed


def repeated_IC(G, nodes_zero, nodes_one, seeds, seeds_type, n_expr, imp_prob, recover_prob = 0, remove = 0):
  zeros_count = []
  ones_count = []
  total_count = []

  for i in range(n_expr):
    impressed = IC(G, seeds, imp_prob, recover_prob = recover_prob, remove = remove)
    total_count.append(len(impressed))

    count_zeros = 0
    count_ones = 0

    for imp in impressed:
      if imp in nodes_zero:
        count_zeros += 1
      elif imp in nodes_one:
        count_ones += 1

    zeros_count.append(count_zeros)
    ones_count.append(count_ones)

  total_imp = np.round(np.mean(total_count), 2)
  total_fraction = np.round(total_imp / len(G.nodes()), 3)

  fraction_zero = np.round(np.mean(zeros_count) / len(nodes_zero), 3)
  fraction_one = np.round(np.mean(ones_count) / len(nodes_one), 3)

  return total_imp, total_fraction, fraction_zero, fraction_one


def train_aae():
    embedding_size = 30
    edges_filename = "/home/andreib/FACT-AI/Crosswalk/data/synth2/synthetic_n500_Pred0.7_Phom0.025_Phet0.001.links"
    attr_filename = "/home/andreib/FACT-AI/Crosswalk/data/synth2/synthetic_n500_Pred0.7_Phom0.025_Phet0.001.attr"

    # Can get `with_f1` or `without_f1`
    first_order_imp = 'no_f1'
    alpha = 0.05

    # 1. Creating the Graph and Getting the Adj Matrix
    G, X, input_G = get_graph_real(edges_filename)
    n = len(G.nodes())

    # 2. Getting seperate lists for seperate communities and the label for each community
    nodes_zero, nodes_one, labels = get_nodes_labels_real(input_G, attr_filename)

    # 3. Getting the idxs suitable for training.
    idxs, idxs_zeros, idxs_ones = get_idxs(n, nodes_zero, nodes_one)

    print("Successfully loaded the graphs")

    # 4. Creating the Embedder
    encoder = build_encoder(embedding_size)
    decoder = build_decoder(embedding_size, n)
    auto_encoder = build_ae(encoder, decoder, n)

    # 5. Creating the Discriminator
    discriminator = build_discriminator(embedding_size)

    # 6. Pretraining the Embedder and the Discriminator
    pre_optimizer_embd = tf.keras.optimizers.Adam()
    pre_optimizer_disc = tf.keras.optimizers.Adam()

    print("Starting training.......")

    pretrain_embd(X, idxs, encoder, decoder, auto_encoder, pre_optimizer_embd, first_order_imp, alpha)
    pretrain_disc(X, idxs_zeros, idxs_ones, encoder, discriminator, pre_optimizer_disc)
    # print('6')

    # # 6-1. Get the pretrain-embeddings
    pre_embds = encoder(X)
    print('pre-training done.')

    # 7. Adversarial Training
    ae_optimizer = tf.keras.optimizers.Adam()
    disc_optimizer = tf.keras.optimizers.Adam()

    adversarial_train(X, idxs_zeros, idxs_ones, encoder, decoder, auto_encoder, discriminator, ae_optimizer, disc_optimizer, first_order_imp, alpha)

    #7-1. Getting the fair-embeddings
    fair_embds = encoder(X)

    print('adversarial training done.')

    N_CLUSs = [4]

    n_seedss = [1, 2, 3, 4, 5, 6, 7, 8, 10]
    # n_seedss = [8]

    # Methods for getting the seeds can be nearest or re-cluster
    strategy = 're-cluster'

    rows = '['
    first = True


    for N_CLUS in N_CLUSs:
        for n_seeds in n_seedss:
            if first:
                first = False
            else:
                rows += ',\n'
            #8. Getting the seeds for the embeddings and baselines
            fair_seeds = get_seeds(N_CLUS, fair_embds, idxs, labels, nodes_zero, nodes_one, strategy, n_seeds)
            pre_seeds = get_seeds(N_CLUS, pre_embds, idxs, labels, nodes_zero, nodes_one, strategy, n_seeds)

            #9. Getting the final results
            total_fair, fair_frac, zero_fair, one_fair = repeated_IC(G, nodes_zero, nodes_one, fair_seeds, 'fair', 2000, 0.01)
            total_pre, pre_frac, zero_pre, one_pre = repeated_IC(G, nodes_zero, nodes_one, pre_seeds, 'pre', 2000, 0.01)

            #10. Building the current row and adding it to the rows.
            row =[ embedding_size ,  N_CLUS,  n_seeds, total_fair, fair_frac, zero_fair, one_fair,  total_pre ,  pre_frac ,  zero_pre ,  one_pre, '\'' + strategy + '\'']
            print(row)

            rows += '[' + ', '.join(map(str, row)) + ']'

        rows += ']'


    print(rows)


if __name__ == "__main__":
    start_time = time.time()
    train_aae()
    end_time = time.time()

    print(f"Execution time is: {end_time - start_time}")