"""
This is the SNF2 example on simulated butterfly datasets. There are 10 different types of butterfly
"""

import numpy as np
import pandas as pd
import snf
from sklearn.cluster import spectral_clustering
from sklearn.metrics import v_measure_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import sys
import os
import argparse

d = os.path.dirname(os.getcwd())
sys.path.insert(0, d)
# from snf2.tsne_deeplearning import tsne_p_deep
from snf2.tsne_deeplearning_gat import tsne_p_deep
from snf2.embedding import tsne_p, project_tsne
from snf2.main import dist2, snf2, kernel_matching, snf2_original
from snf2.util import data_indexing

# Hyperparameters
parser = argparse.ArgumentParser("SNF2 on butterfly dataset!")
parser.add_argument("--neighbor_size", type=int, default=20)
parser.add_argument("--embedding_dims", type=int, default=50)
parser.add_argument("--fusing_iteration", type=int, default=20)
parser.add_argument("--normalization_factor", type=int, default=1.0)
parser.add_argument("--alighment_epochs", type=int, default=1000)
parser.add_argument("--beta", type=float, default=1.0)
parser.add_argument("--mu", type=float, default=0.5)

args = parser.parse_args()

# read the data
testdata_dir = os.path.join(d, "data/subsample_new/subsample_0.8")
w1_ = os.path.join(testdata_dir, "w1.csv")
w2_ = os.path.join(testdata_dir, "w2.csv")
w1 = pd.read_csv(w1_, index_col=0)
w2 = pd.read_csv(w2_, index_col=0)

label = ["label"]
w1_label = w1[label]
w2_label = w2[label]
wcom_label = w1_label.filter(regex="^common_", axis=0)
w1.drop(label, axis=1, inplace=True)
w2.drop(label, axis=1, inplace=True)
wall_label = pd.concat([w1_label, w2_label], axis=0)
wall_label = wall_label[~wall_label.index.duplicated(keep="first")]



"""
    Step2 : Use SNF2 to fuse not only the common samples, but also the unique samples
"""

Dist1 = dist2(w1.values, w1.values)
Dist2 = dist2(w2.values, w2.values)

S1 = snf.compute.affinity_matrix(Dist1, K=args.neighbor_size, mu=args.mu)
S2 = snf.compute.affinity_matrix(Dist2, K=args.neighbor_size, mu=args.mu)

labels_s1 = spectral_clustering(S1, n_clusters=10)
score1 = v_measure_score(w1_label["label"].tolist(), labels_s1)
print("Before diffusion for full {} p1 NMI score: ".format(len(labels_s1)), score1)

labels_s2 = spectral_clustering(S2, n_clusters=10)
score2 = v_measure_score(w2_label["label"].tolist(), labels_s2)
print("Before diffusion for full {} p2 NMI score:".format(len(labels_s2)), score2)

# Do SNF2 diffusion
(
    dicts_common,
    dicts_commonIndex,
    dict_sampleToIndexs,
    dicts_unique,
    original_order,
) = data_indexing([w1, w2])
S1_df = pd.DataFrame(data=S1, index=original_order[0], columns=original_order[0])
S2_df = pd.DataFrame(data=S2, index=original_order[1], columns=original_order[1])

fused_networks = snf2(
    args,
    [S1_df, S2_df],
    dicts_common=dicts_common,
    dicts_unique=dicts_unique,
    original_order=original_order,
)

S1_fused = fused_networks[0]
S2_fused = fused_networks[1]

labels_s1 = spectral_clustering(S1_fused.values, n_clusters=10)
score1 = v_measure_score(w1_label["label"].tolist(), labels_s1)
print("Before diffusion for full {} p1 NMI score: ".format(len(labels_s1)), score1)

labels_s2 = spectral_clustering(S2_fused.values, n_clusters=10)
score2 = v_measure_score(w2_label["label"].tolist(), labels_s2)
print("Before diffusion for full {} p2 NMI score:".format(len(labels_s2)), score2)

"""
    Step3 : Use the t-SNE to extract embedding vectors from similarity networks
"""
# w1_tsne = tsne_p(S1_fused.values, no_dims=20)
# w2_tsne = tsne_p(S2_fused.values, no_dims=20)
# np.savetxt("/Users/mashihao/Desktop/SNF2/data/w1_tsne.csv", w1_tsne, delimiter=",")
# np.savetxt("/Users/mashihao/Desktop/SNF2/data/w2_tsne.csv", w2_tsne, delimiter=",")

# load t-sne
# tsne_w1 = os.path.join(testdata_dir, "w1_tsne.csv")
# tsne_w2 = os.path.join(testdata_dir, "w2_tsne.csv")
# w1_tsne = np.loadtxt(tsne_w1, delimiter=",")
# w2_tsne = np.loadtxt(tsne_w2, delimiter=",")


S_final = tsne_p_deep(
    args,
    dicts_commonIndex,
    dict_sampleToIndexs,
    [S1_fused.values, S2_fused.values],
)


"""
    Step4 : Iterative matching to integrate extracted embedding vectors into one network
"""

# # relabel the sample ID to the t-SNE embedding vectors
# embed_w1 = pd.DataFrame(data=w1_tsne, index=original_order[0])
# embed_w2 = pd.DataFrame(data=w2_tsne, index=original_order[1])


# S_final = kernel_matching(
#     [embed_w1, embed_w2],
#     dicts_common=dicts_common,
#     dicts_unique=dicts_unique,
#     alpha=0.1,
#     matching_iter=50,
# )
# S_final = S_final.reindex((wall_label.index.tolist()), axis=0)

# Dist_final = dist2(S_final.values, S_final.values)
S_final_df = pd.DataFrame(data=S_final, index=dict_sampleToIndexs.keys())
S_final_df = S_final_df.reindex((wall_label.index.tolist()), axis=0)

Dist_final = dist2(S_final_df.values, S_final_df.values)

Wall_final = snf.compute.affinity_matrix(Dist_final, K=args.neighbor_size, mu=args.mu)

labels_final = spectral_clustering(Wall_final, n_clusters=10)
score = v_measure_score(wall_label["label"].tolist(), labels_final)
print("SNF2 + kernel matching for clustering union 832 samples NMI score:", score)


# # lets do some ploting
# X_embedded = TSNE(n_components=2).fit_transform(S_final)
# plt.scatter(
#     X_embedded[:, 0],
#     X_embedded[:, 1],
#     c=wall_label["label"].tolist(),
#     s=1.5,
#     cmap="Spectral",
# )
# plt.title("t-SNE visualization of union 1332 butterfly")
# save_path = os.path.join(testdata_dir, "t-SNE_butterfly.png")
# plt.savefig(save_path)
# print("Save visualization at {}".format(save_path))
