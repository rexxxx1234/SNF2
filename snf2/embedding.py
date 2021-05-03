#
#  tsne_p.py
#
# Implementation of t-SNE in Python. The implementation was tested on Python
# 2.7.10, and it requires a working installation of NumPy.
#
#
#  Created by Laurens van der Maaten on 20-12-08. Modify by Rex Ma on 11-06-19
#  Copyright (c) 2008 Tilburg University. All rights reserved.

import numpy as np
from snf2.Model import model

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os


def tsne_p(P=np.array([]), no_dims=50, initial_dims=50, perplexity=30.0):
    """
    Runs t-SNE on the dataset in the NxN matrix P to extract embedding vectors
    to no_dims dimensions. The syntaxis of the function is
    `Y = tsne.tsne_p(P, no_dims, perplexity), where P is an NxN NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array P should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    print("Start applying t-SNE extraction!")

    # Initialize variables
    (n, d) = P.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Make sure P-values are set properly
    np.fill_diagonal(P, 0)  # set diagonal to zero
    P = P + np.transpose(P)  # symmetrize P-values
    P = P / np.sum(P)  # make sure P-values sum to one
    P = P * 4.0  # early exaggeration
    P = np.maximum(P, 1e-12)

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = -2.0 * np.dot(Y, Y.T)
        num = 1.0 / (1.0 + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.0
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = np.sum(
                np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0
            )

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.0) != (iY > 0.0)) + (gains * 0.8) * (
            (dY > 0.0) == (iY > 0.0)
        )
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (iter + 1) % 100 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.0

    # Return solution
    return Y


def init_model(net, device, restore):
    if restore is not None and os.path.exits(restore):
        net.load_state_dict(torch.load(restore))
        net.restored = True
        print("Restore model from: {}".format(os.path.abspath(restore)))

    else:
        pass

    if torch.cuda.is_available():
        cudnn.benchmark = True
        net.to(device)

    return net


def neg_square_dists(X):
    sum_X = torch.sum(X * X, 1)
    tmp = torch.add(-2 * X.mm(torch.transpose(X, 1, 0)), sum_X)
    D = torch.add(torch.transpose(tmp, 1, 0), sum_X)
    return -D


def Q_tsne(Y):
    distances = neg_square_dists(Y)
    inv_distances = torch.pow(1.0 - distances, -1)
    inv_distances = inv_distances - torch.diag(inv_distances.diag(0))
    inv_distances = inv_distances + 1e-15
    return inv_distances / torch.sum(inv_distances)


def project_tsne(dataset, P_joint, num_com, no_dims=50):

    print("---------------------------------")
    print("Begin finding the embedded space")

    dataset_num = len(dataset)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lr = 0.001
    batch_size = 100
    epoch_DNN = 1000
    log_DNN = 50
    beta = 10

    # get data dims
    col = []
    row = []
    print("Shape of Raw data")
    for i in range(dataset_num):
        row.append(np.shape(dataset[i])[0])
        col.append(np.shape(dataset[i])[1])
        print("Dataset {}:".format(i), np.shape(dataset[i]))

    for i in range(dataset_num):
        P_joint[i] = torch.from_numpy(P_joint[i]).float().to(device)
        dataset[i] = torch.from_numpy(dataset[i]).float().to(device)

    net = model(col, no_dims)
    Project_DNN = init_model(net, device, restore=None)

    optimizer = optim.Adam(Project_DNN.parameters(), lr=lr)
    c_mse = nn.MSELoss()
    Project_DNN.train()

    for epoch in range(epoch_DNN):
        len_dataloader = np.int(np.max(row) / batch_size)
        if len_dataloader == 0:
            len_dataloader = 1
            batch_size = np.max(row)
        for step in range(len_dataloader):
            KL_loss = []
            for i in range(dataset_num):
                random_batch = np.random.randint(0, row[i], batch_size)
                data = dataset[i][random_batch]
                P_tmp = torch.zeros([batch_size, batch_size]).to(device)
                for j in range(batch_size):
                    P_tmp[j] = P_joint[i][random_batch[j], random_batch]
                P_tmp = P_tmp / torch.sum(P_tmp)
                low_dim_data = Project_DNN(data, i)
                Q_joint = Q_tsne(low_dim_data)

                ## loss of structure preserving
                KL_loss.append(torch.sum(P_tmp * torch.log(P_tmp / Q_joint)))

            # import ipdb
            # ipdb.set_trace()

            ## loss of structure matching
            feature_loss = np.array(0)
            feature_loss = torch.from_numpy(feature_loss).to(device).float()
            for i in range(dataset_num - 1):
                low_dim_set1 = Project_DNN(
                    dataset[i][
                        0:num_com,
                    ],
                    i,
                )
                low_dim_set2 = Project_DNN(
                    dataset[dataset_num - 1][
                        0:num_com,
                    ],
                    len(dataset) - 1,
                )
                feature_loss += c_mse(low_dim_set1, low_dim_set2)

            loss = beta * feature_loss
            for i in range(dataset_num):
                loss += KL_loss[i]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % log_DNN == 0:
            print(
                "epoch:[{:d}/{}]: loss:{:4f}, align_loss:{:4f}".format(
                    epoch + 1, epoch_DNN, loss.data.item(), feature_loss.data.item()
                )
            )

    integrated_data = []
    for i in range(dataset_num):
        integrated_data.append(Project_DNN(dataset[i], i))
        integrated_data[i] = integrated_data[i].detach().cpu().numpy()
    print("Done")
    return integrated_data
