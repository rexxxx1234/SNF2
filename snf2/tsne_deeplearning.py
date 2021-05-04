import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np


def tsne_loss(P, activations):
    n = activations.size(0)
    alpha = 1
    eps = 1e-15
    sum_act = torch.sum(torch.pow(activations, 2), 1)
    Q = sum_act + sum_act.view([-1, 1]) - 2 * torch.matmul(activations, torch.transpose(activations, 0, 1))
    Q = Q / alpha
    Q = torch.pow(1 + Q, -(alpha + 1) / 2)
    Q = Q * autograd.Variable(1 - torch.eye(n), requires_grad=False)
    Q = Q / torch.sum(Q)
#     Q = torch.clamp(Q, min=eps)
    C = torch.log((P + eps) / (Q + eps))
    C = torch.sum(P * C)
    return C


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 50 epochs"""
    lr = 0.1 * (0.1 ** (epoch // 150))
    lr = max(lr, 1e-3)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def tsne_p_deep(P=np.array([]), data, no_dims=50, perplexity=30.0):
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
    data = torch.from_numpy(data).float().to(device)

    D_in, fc1, fc2, fc3, D_out = 500, 1024, 512, 256, 20

    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, fc1), torch.nn.ReLU(),
        torch.nn.Linear(fc1, fc2), torch.nn.ReLU(),
        torch.nn.Linear(fc2, fc3), torch.nn.ReLU(),
        torch.nn.Linear(fc3, D_out)
    )
    model.to(device)

    n = data.shape[0]
    batch_size = 200
    batch_size = min(batch_size, n)
    batch_count = int(n / batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
    for epoch in range(450):      
        adjust_learning_rate(optimizer, epoch)
        for i, start in enumerate(range(0, n - batch_size + 1, batch_size)):   
            curX = data[start:start+batch_size]  
            curP = P[start:start+batch_size, start:start+batch_size]
            #x_var = autograd.Variable(torch.Tensor(curX), requires_grad=False)
            #P_var = autograd.Variable(torch.Tensor(curP), requires_grad=False)
            y_pred = model(x_var)
            loss = tsne_loss(P_var, y_pred)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print('epoch {}: loss {}'.format(epoch, loss.data.item()))

    embeddings = model(data).detach().cpu().numpy()
    print("Done")
return embeddings