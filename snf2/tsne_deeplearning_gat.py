import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from snf2.GAT import model
from snf.compute import _find_dominate_set
import numpy as np
import networkx as nx
import dgl
import time


def tsne_loss(P, activations):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n = activations.size(0)
    alpha = 1
    eps = 1e-12
    sum_act = torch.sum(torch.pow(activations, 2), 1)
    Q = (
        sum_act
        + sum_act.view([-1, 1])
        - 2 * torch.matmul(activations, torch.transpose(activations, 0, 1))
    )
    Q = Q / alpha
    Q = torch.pow(1 + Q, -(alpha + 1) / 2)
    Q = Q * autograd.Variable(1 - torch.eye(n), requires_grad=False).to(device)
    Q = Q / torch.sum(Q)
    Q = torch.clamp(Q, min=eps)
    C = torch.log((P + eps) / (Q + eps))
    C = torch.sum(P * C)
    return C


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 0.1 every 100 epochs"""
    lr = 0.1 * (0.1 ** (epoch // 200))
    lr = max(lr, 1e-4)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


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


def P_preprocess(P):
    # Make sure P-values are set properly
    np.fill_diagonal(P, 0)  # set diagonal to zero
    P = P + np.transpose(P)  # symmetrize P-values
    P = P / np.sum(P)  # make sure P-values sum to one
    P = P * 4.0  # early exaggeration
    P = np.maximum(P, 1e-12)
    return P


def tsne_p_deep(args, dicts_commonIndex, dict_sampleToIndexs, dataset, P=np.array([])):
    """
    Runs t-SNE on the dataset in the NxN matrix P to extract embedding vectors
    to no_dims dimensions.
    """
    # Check inputs
    if isinstance(args.embedding_dims, float):
        print("Error: array P should have type float.")
        return -1
    if round(args.embedding_dims) != args.embedding_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    print("Start applying deep-learning based t-SNE extraction!")
    start_time = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset_num = len(dataset)
    G = []
    feature_dims = []
    for i in range(dataset_num):
        # get dataset dimentions
        feature_dims.append(np.shape(dataset[i])[1])
        print("Dataset {}:".format(i), np.shape(dataset[i]))

        dataset[i] = torch.from_numpy(dataset[i]).float().to(device)

        # construct DGL graph
        temp = _find_dominate_set(P[i], K=args.neighbor_size)
        #temp = _find_dominate_set(P[i], K=10)
        g_nx = nx.from_numpy_matrix(temp)
        g_dgl = dgl.DGLGraph(g_nx)
        g_dgl = g_dgl.to(device)
        G.append(g_dgl)

        # preprocess similarity matrix for t-sne loss
        P[i] = P_preprocess(P[i])
        P[i] = torch.from_numpy(P[i]).float().to(device)

    net = model(G, feature_dims, args.embedding_dims)
    Project_DNN = init_model(net, device, restore=None)
    Project_DNN.train()

    optimizer = torch.optim.Adam(Project_DNN.parameters(), lr=1e-3)
    c_mse = nn.MSELoss()

    for epoch in range(args.alighment_epochs):
        adjust_learning_rate(optimizer, epoch)

        loss = 0
        embeddings = []

        # KL loss for each network
        for i in range(dataset_num):
            X_embedding = Project_DNN(dataset[i], i)
            embeddings.append(X_embedding)
            kl_loss = tsne_loss(P[i], X_embedding)
            loss += kl_loss

        # pairwise alignment loss between each pair of networks
        alignment_loss = np.array(0)
        alignment_loss = torch.from_numpy(alignment_loss).to(device).float()

        for i in range(dataset_num - 1):
            for j in range(i + 1, dataset_num):
                low_dim_set1 = embeddings[i][dicts_commonIndex[(i, j)]]
                low_dim_set2 = embeddings[j][dicts_commonIndex[(j, i)]]
                alignment_loss += c_mse(low_dim_set1, low_dim_set2)

        loss += args.beta * alignment_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch == 100:
            for i in range(dataset_num):
                P[i] = P[i] / 4.0

        if (epoch) % 10 == 0:
            print(
                "epoch {}: loss {}, align_loss:{:4f}".format(
                    epoch, loss.data.item(), alignment_loss.data.item()
                )
            )

    # get the final embeddings for all samples
    embeddings = []
    for i in range(dataset_num):
        embeddings.append(Project_DNN(dataset[i], i))
        embeddings[i] = embeddings[i].detach().cpu().numpy()

    final_embedding = np.array([]).reshape(0, args.embedding_dims)
    for key in dict_sampleToIndexs:
        sample_embedding = np.zeros((1, args.embedding_dims))

        for (dataset, index) in dict_sampleToIndexs[key]:
            sample_embedding += embeddings[dataset][index]
        sample_embedding /= len(dict_sampleToIndexs[key])

        final_embedding = np.concatenate((final_embedding, sample_embedding), axis=0)

    end_time = time.time()
    print("Manifold alignment ends! Times: {}s".format(end_time - start_time))

    return final_embedding
