import numpy as np
import pandas as pd
import time
from scipy import sparse, stats
from snf.compute import _flatten, _find_dominate_set, _B0_normalized
from sklearn.utils.validation import (
    check_array,
    check_symmetric,
    check_consistent_length,
)


def dist2(X, C):
    """
    Description: Computes the Euclidean distances between all pairs of data point given

    Usage: dist2(X, C)
    X: A data matrix where each row is a different data point
    C: A data matrix where each row is a different data point. If this matrix is the same as X,
    pairwise distances for all data points in X are computed.

    Return: Returns an N x M matrix where N is the number of rows in X and M is the number of rows in C.

    Author: Dr. Anna Goldenberg, Bo Wang, Aziz Mezlini, Feyyaz Demir
    Python Version Rewrite: Rex Ma

    Examples:
        # Data1 is of size n x d_1, where n is the number of patients, d_1 is the number of genes,
        # Data2 is of size n x d_2, where n is the number of patients, d_2 is the number of methylation
        Dist1 = dist2(Data1, Data1)
        Dist2 = dist2(Data2, Data2)
    """
    ndata = X.shape[0]
    ncentres = C.shape[0]

    sumsqX = np.sum(X * X, axis=1)
    sumsqC = np.sum(C * C, axis=1)

    XC = 2 * (np.matmul(X, np.transpose(C)))

    res = (
        np.transpose(np.reshape(np.tile(sumsqX, ncentres), (ncentres, ndata)))
        + np.reshape(np.tile(sumsqC, ndata), (ndata, ncentres))
        - XC
    )

    return res


def _check_SNF2_inputs(aff):
    """
    Confirms inputs to SNF2 are appropriate

    Parameters
    ----------
    aff : `m`-list of (N x N) array_like
        Input similarity arrays. All arrays should be square but no need to be equal size.
    """

    prep = []
    for a in _flatten(aff):
        ac = check_array(a, force_all_finite=True, copy=True)
        prep.append(check_symmetric(ac, raise_warning=False))

    return prep


def _stable_normalized(W):
    """
    Adds `alpha` to the diagonal of `W`

    Parameters
    ----------
    W : (N, N) array_like
        Similarity array from SNF

    Returns
    -------
    W : (N, N) np.ndarray
        Stable-normalized similiarity array
    """

    # add `alpha` to the diagonal and symmetrize `W`
    rowSum = np.sum(W, 1) - np.diag(W)
    rowSum[rowSum == 0] = 1

    W = W / (2 * rowSum)
    np.fill_diagonal(W, 0.5)
    W = check_symmetric(W, raise_warning=False)

    return W


def _stable_normalized_pd(W):
    """
    Adds `alpha` to the diagonal of pandas dataframe `W`

    Parameters
    ----------
    W : (N, N) array_like
        Similarity array from SNF

    Returns
    -------
    W : (N, N) np.ndarray
        Stable-normalized similiarity array
    """

    # add `alpha` to the diagonal and symmetrize `W`
    rowSum = np.sum(W, 1) - np.diag(W)
    rowSum[rowSum == 0] = 1

    W = W / (2 * rowSum)
    np.fill_diagonal(W.values, 0.5)
    W = check_symmetric(W, raise_warning=False)

    return W


def handle_zeros_in_scale(scale, copy=True):
    """Makes sure that whenever scale is zero, we handle it correctly.
    This happens in most scalers when we have constant features.
    Adapted from sklearn.preprocessing.data"""

    # if we are fitting on 1D arrays, scale might be a scalar
    if np.isscalar(scale):
        if scale == 0.0:
            scale = 1.0
        return scale
    elif isinstance(scale, np.ndarray):
        if copy:
            # New array to avoid side-effects
            scale = scale.copy()
        scale[scale == 0.0] = 1.0
    return scale


def affinity_matrix_mixdimen(dist, *, K=20, mu=0.5):
    # check inputs
    dist = check_array(dist, force_all_finite=False)

    # get mask for potential NaN values and set diagonals zero
    mask = np.isnan(dist)

    # sort array and get average distance to K nearest neighbors
    T = np.sort(dist, axis=1)
    TT = np.vstack(T[:, 1 : K + 1].mean(axis=1) + np.spacing(1))

    # sort array and get average distance to K nearest neighbors
    Y = np.sort(dist, axis=0)
    YY = np.vstack(Y[1 : K + 1, :].mean(axis=0) + np.spacing(1))

    # compute sigma (see equation in Notes)
    sigma = (TT + YY.T + dist) / 3
    msigma = np.ma.array(sigma, mask=mask)  # mask for NaN
    sigma = sigma * np.ma.greater(msigma, np.spacing(1)).data + np.spacing(1)

    # get probability density function with scale = mu*sigma and symmetrize
    scale = (mu * np.nan_to_num(sigma)) + mask
    W = stats.norm.pdf(np.nan_to_num(dist), loc=0, scale=scale)
    W[mask] = np.nan

    return W


def snf2_np(*aff, numofCom, K=20, t=20, alpha=1.0):
    """
    Performs Similarity Network Fusion on `aff` matrices

    Parameters
    ----------
    *aff : (N, N) array_like
        Input similarity arrays; all arrays should be square but no need to be equal size.
        Note: these arrays must have all the common samples appeared first in the matrix

    numofCom: int, required
        Number of common samples across all the matrices

    K : (0, N) int, optional
        Hyperparameter normalization factor for scaling. Default: 20

    t : int, optional
        Number of iterations to perform information swapping. Default: 20

    alpha : (0, 1) float, optional
        Hyperparameter normalization factor for scaling. Default: 1.0

    Returns
    -------
    W: (N, N) Ouputs similarity arrays
        Fused similarity networks of input arrays
    """
    print("Start applying diffusion!")
    aff = _check_SNF2_inputs(aff)
    newW = [0] * len(aff)
    aff_com = [0] * len(aff)

    # First, normalize different networks to avoid scale problems
    for n, mat in enumerate(aff):
        # normalize affinity matrix based on strength of edges
        # mat = mat / np.nansum(mat, axis=1, keepdims=True)
        mat = _stable_normalized(mat)
        aff[n] = check_symmetric(mat, raise_warning=False)
        aff_com[n] = aff[n][0:numofCom, :][:, 0:numofCom]
        # apply KNN threshold to normalized affinity matrix
        # We need to crop the intersecting samples from newW matrices
        newW[n] = _find_dominate_set(aff[n], int(K))
        newW[n] = newW[n][:, 0:numofCom]

    # take sum of all normalized (not thresholded) affinity matrices of the intersections part
    Wsum = np.nansum(aff_com, axis=0)

    # get number of modalities informing each subject x subject affinity
    n_aff = len(aff_com) - np.sum([np.isnan(a) for a in aff_com], axis=0)

    for iteration in range(t):
        for n, mat in enumerate(aff):
            # temporarily convert nans to 0 to avoid propagation errors
            nzW = np.nan_to_num(newW[n])
            mat = mat[0:numofCom, :][:, 0:numofCom]
            aw = np.nan_to_num(mat)
            # propagate `Wsum` through masked affinity matrix (`nzW`)
            aff0 = np.matmul(
                np.matmul(nzW, (Wsum - aw) / (n_aff - 1)), nzW.T
            )  # TODO: / by 0
            # ensure diagonal retains highest similarity
            aff[n] = _B0_normalized(aff0, alpha=alpha)
            aff_com[n] = aff[n][0:numofCom, :][:, 0:numofCom]

        # compute updated sum of normalized affinity matrices
        Wsum = np.nansum(aff_com, axis=0)

    for n, mat in enumerate(aff):
        mat = _stable_normalized(mat)
        aff[n] = check_symmetric(mat, raise_warning=False)

    return aff


def snf2(args, aff, dicts_common, dicts_unique, original_order):
    """
    Performs Similarity Network Fusion on `aff` matrices

    Parameters
    ----------
    aff : (N, N) pandas dataframe
        Input similarity arrays; all arrays should be square but no need to be equal size.

    dicts_common: dictionaries, required
        Dictionaries for getting common samples from different views
        Example: dicts_common[(0, 1)] == dicts_common[(1, 0)], meaning the common patients between view 1&2

    dicts_unique: dictionaries, required
        Dictionaries for getting unique samples for different views
        Example: dicts_unique[(0, 1)], meaning the unique samples from view 1 that are not in view 2
                 dicts_unique[(1, 0)], meaning the unique samples from view 2 that are not in view 1

    original_order: lists, required
        The original order of each view

    K : (0, N) int, optional
        Hyperparameter normalization factor for scaling. Default: 20

    t : int, optional
        Number of iterations to perform information swapping. Default: 20

    alpha : (0, 1) float, optional
        Hyperparameter normalization factor for scaling. Default: 1.0

    Returns
    -------
    W: (N, N) Ouputs similarity arrays
        Fused similarity networks of input arrays
    """

    print("Start applying diffusion!")
    start_time = time.time()

    newW = [0] * len(aff)

    # First, normalize different networks to avoid scale problems, it is compatible with pandas dataframe
    for n, mat in enumerate(aff):
        # normalize affinity matrix based on strength of edges
        # mat = mat / np.nansum(mat, axis=1, keepdims=True)
        mat = _stable_normalized_pd(mat)
        aff[n] = check_symmetric(mat, raise_warning=False)

        # apply KNN threshold to normalized affinity matrix
        # We need to crop the intersecting samples from newW matrices
        newW[n] = _find_dominate_set(aff[n], int(args.neighbor_size))

    for iteration in range(args.fusing_iteration):
        for n, mat in enumerate(aff):
            # temporarily convert nans to 0 to avoid propagation errors
            nzW = newW[n]  # TODO: not sure this is a deep copy or not

            # Your goal is to update aff[n], but it is the average of all the defused matrices.
            # Make a copy of add[n], and set it to 0
            aff0_copy = aff[n].copy()
            for col in aff0_copy.columns:
                aff0_copy[col].values[:] = 0

            for j, mat_tofuse in enumerate(aff):
                if n == j:
                    continue

                # reorder nzW and mat_tofuse to have the correct order of samples
                nzW = nzW.reindex(
                    (sorted(dicts_common[(n, j)]) + sorted(dicts_unique[(n, j)])),
                    axis=1,
                )
                nzW = nzW.reindex(
                    (sorted(dicts_common[(n, j)]) + sorted(dicts_unique[(n, j)])),
                    axis=0,
                )
                mat_tofuse = mat_tofuse.reindex(
                    (sorted(dicts_common[(j, n)]) + sorted(dicts_unique[(j, n)])),
                    axis=1,
                )
                mat_tofuse = mat_tofuse.reindex(
                    (sorted(dicts_common[(j, n)]) + sorted(dicts_unique[(j, n)])),
                    axis=0,
                )

                # Next, let's crop mat and mat_tofuse
                num_common = len(dicts_common[(n, j)])

                to_drop_nzW = nzW.columns[num_common : nzW.shape[1]].values.tolist()
                nzW_crop = nzW.drop(to_drop_nzW, axis=1)
                nzW_crop_T = np.transpose(nzW_crop)

                to_drop_mat = mat_tofuse.columns[
                    num_common : mat_tofuse.shape[1]
                ].values.tolist()
                mat_tofuse_crop = mat_tofuse.drop(to_drop_mat, axis=1)
                mat_tofuse_crop = mat_tofuse_crop.drop(to_drop_mat, axis=0)

                # Now we are ready to do the diffusion
                aff0_temp = nzW_crop.dot(
                    mat_tofuse_crop.dot(nzW_crop_T)
                )  # Matmul is not working, but .dot() is good

                aff0_temp = _B0_normalized(aff0_temp, alpha=args.normalization_factor)

                # Reorder back, and then add the aff0_copy
                aff0_temp = aff0_temp.reindex(original_order[n], axis=1)
                aff0_temp = aff0_temp.reindex(original_order[n], axis=0)
                aff0_copy = np.add(aff0_temp, aff0_copy)

            # Get the average then update the aff[n]
            aff[n] = np.divide(aff0_copy, len(aff) - 1)

    for n, mat in enumerate(aff):
        mat = _stable_normalized_pd(mat)
        aff[n] = check_symmetric(mat, raise_warning=False)

    end_time = time.time()
    print("Diffusion ends! Times: {}s".format(end_time - start_time))
    return aff


def kernel_matching(aff, dicts_common, dicts_unique, alpha=0.1, matching_iter=25):
    """
    Performs common-unique samples integration to get the final union similarity matrix

    Parameters
    ----------
    aff : (N, M) pandas dataframes
        Input embedding vectors from all views.

    dicts_common: dictionaries, required
        Dictionaries for getting common samples from different views
        Example: dicts_common[(0, 1)] == dicts_common[(1, 0)], meaning the common patients between view 1&2

    dicts_unique: dictionaries, required
        Dictionaries for getting unique samples for different views
        Example: dicts_unique[(0, 1)], meaning the unique samples from view 1 that are not in view 2
                 dicts_unique[(1, 0)], meaning the unique samples from view 2 that are not in view 1

    Returns
    -------
    W: (U, M) Output the integrated embedding vector for all views
    """
    print("Start iterative final matching!")
    start_time = time.time()

    # sanity check
    if len(aff) == 1:
        print("Only input one view, return the view itself")
        return aff[0]
    elif len(aff) == 0:
        print("Error: no input view, return nothing")
        return None

    # First, let's find the integration order based on the number of common sample
    # between each view.
    integrate_order = []
    for k in sorted(dicts_common, key=lambda k: len(dicts_common[k]), reverse=True):
        if k[0] not in integrate_order:
            integrate_order.append(k[0])
        if k[1] not in integrate_order:
            integrate_order.append(k[1])

    final_view = pd.DataFrame()
    for iter, n_view in enumerate(integrate_order):
        if iter == 0:
            final_view = aff[n_view]
            continue

        # First find common and unique samples between reference view and to_match view
        tomatch_view = aff[n_view]
        sample_ref = list(np.transpose(final_view))
        sample_tomatch = list(np.transpose(tomatch_view))
        commonList = list(set(sample_ref).intersection(sample_tomatch))
        unique_ref = list(set(sample_ref).symmetric_difference(commonList))
        unique_tomatch = list(set(sample_tomatch).symmetric_difference(commonList))

        final_view = final_view.reindex((commonList + unique_ref), axis=0)
        tomatch_view = tomatch_view.reindex((commonList + unique_tomatch), axis=0)
        final_view_np = final_view.values
        tomatch_view_np = tomatch_view.values

        numofcom = len(commonList)
        com_ind = np.arange(numofcom)

        # use snf kernel
        tomatch_dist = dist2(tomatch_view_np, tomatch_view_np[com_ind, :])
        tomatch_weights = affinity_matrix_mixdimen(tomatch_dist, K=20, mu=0.5)
        tomatch_weights = _find_dominate_set(tomatch_weights, K=10)

        denom = np.sum(tomatch_weights, axis=1)
        denom = handle_zeros_in_scale(denom, copy=False)
        denom = denom[:, np.newaxis]

        for i in range(0, matching_iter):
            w_tomatch = tomatch_view_np[com_ind, :].copy()
            w_ref = final_view_np[com_ind, :].copy()
            tomatch_view_np = (
                tomatch_view_np
                + alpha * np.dot(tomatch_weights, (w_ref - w_tomatch)) / denom
            )

        # now lets match 2 view together
        tomatch_view_uni = tomatch_view_np[numofcom : tomatch_view_np.shape[0], :]
        final_view_uni = final_view_np[numofcom : final_view_np.shape[0], :]
        final_view_mix = np.concatenate(
            (
                np.add(tomatch_view_np[0:numofcom, :], final_view_np[0:numofcom, :])
                / 2,
                final_view_uni,
                tomatch_view_uni,
            )
        )
        final_view = pd.DataFrame(
            data=final_view_mix, index=(commonList + unique_ref + unique_tomatch)
        )

    end_time = time.time()
    print("Matching ends! Times: {}s".format(end_time - start_time))

    return final_view
