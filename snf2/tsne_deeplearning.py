import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from snf2.Model import model
import numpy as np


def tsne_loss(P, activations):
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
    Q = Q * autograd.Variable(1 - torch.eye(n), requires_grad=False)
    Q = Q / torch.sum(Q)
    Q = torch.clamp(Q, min=eps)
    C = torch.log((P + eps) / (Q + eps))
    C = torch.sum(P * C)
    return C


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 50 epochs"""
    lr = 0.1 * (0.1 ** (epoch // 100))
    lr = max(lr, 1e-3)
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


def tsne_p_deep(data, P=np.array([]), no_dims=50, perplexity=30.0):
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

    print("Start applying deep-learning based t-SNE extraction!")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Make sure P-values are set properly
    np.fill_diagonal(P, 0)  # set diagonal to zero
    P = P + np.transpose(P)  # symmetrize P-values
    P = P / np.sum(P)  # make sure P-values sum to one
    P = P * 4.0  # early exaggeration
    P = np.maximum(P, 1e-12)
    P = torch.from_numpy(P).float().to(device)

    col = []
    dataset_num = len(data)
    for i in range(dataset_num):
        col.append(np.shape(data[i])[1])
        print("Dataset {}:".format(i), np.shape(data[i]))

    for i in range(dataset_num):
        data[i] = torch.from_numpy(data[i]).float().to(device)

    net = model(col, no_dims)
    Project_DNN = init_model(net, device, restore=None)

    n = data[0].shape[0]
    batch_size = 1500
    batch_size = min(batch_size, n)
    batch_count = int(n / batch_size)

    optimizer = torch.optim.Adam(Project_DNN.parameters(), lr=1e-1)
    for epoch in range(1000):
        adjust_learning_rate(optimizer, epoch)
        epoch_loss = 0
        for i, start in enumerate(range(0, n - batch_size + 1, batch_size)):
            curX = data[0][start : start + batch_size]
            curP = P[start : start + batch_size, start : start + batch_size]

            x_var = autograd.Variable(torch.Tensor(curX), requires_grad=False)
            P_var = autograd.Variable(torch.Tensor(curP), requires_grad=False)

            y_pred = Project_DNN(x_var, 0)
            loss = tsne_loss(P_var, y_pred)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.data.item()

        if (epoch) % 50 == 0:
            print("epoch {}: loss {}".format(epoch, epoch_loss))

        if epoch == 100:
            P = P / 4.0

    embeddings = Project_DNN(data[0], 0).detach().cpu().numpy()
    print("Done")
    return embeddings
