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
    np.fill_diagonal(P, 0)     # set diagonal to zero
    P = P + np.transpose(P)        # symmetrize P-values
    P = P / np.sum(P)              # make sure P-values sum to one
    P = P * 4.					   # early exaggeration
    P = np.maximum(P, 1e-12)

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
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
            P = P / 4.

    # Return solution
    return Y


