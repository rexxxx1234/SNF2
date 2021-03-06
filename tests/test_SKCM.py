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
from snf2.tsne_deeplearning_gat import tsne_p_deep
from snf2.embedding import tsne_p
from snf2.main import dist2, snf2, kernel_matching
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
#testdata_dir = os.path.join(d, "data/snf2_cancers/SKCM")
testdata_dir = "/scratch/gobi2/rexma/snf2_cancers/SKCM"
cnv_ = os.path.join(testdata_dir, "cnv_367x24776.csv")
meth_ = os.path.join(testdata_dir, "meth_470x16048.csv")
mirna_ = os.path.join(testdata_dir, "mirna_448x1046.csv")
rnaseq_ = os.path.join(testdata_dir, "rnaseq_469x20531.csv")
rppa_ = os.path.join(testdata_dir, "rppa_353x195.csv")


cnv = pd.read_csv(cnv_, index_col=0)
meth = pd.read_csv(meth_, index_col=0)
mirna = pd.read_csv(mirna_, index_col=0)
rnaseq = pd.read_csv(rnaseq_, index_col=0)
rppa = pd.read_csv(rppa_, index_col=0)
print("finish loading data!")

# data indexing
(
    dicts_common,
    dicts_commonIndex,
    dict_sampleToIndexs,
    dicts_unique,
    original_order,
) = data_indexing(
    [cnv, meth, mirna, rnaseq, rppa]
)
print("finish indexing data!")

# build similarity networks for each motality
dist_cnv = dist2(cnv.values, cnv.values)
dist_meth = dist2(meth.values, meth.values)
dist_mirna = dist2(mirna.values, mirna.values)
dist_rnaseq = dist2(rnaseq.values, rnaseq.values)
dist_rppa = dist2(rppa.values, rppa.values)

S1_cnv = snf.compute.affinity_matrix(dist_cnv, K=args.neighbor_size, mu=args.mu)
S2_meth = snf.compute.affinity_matrix(dist_meth, K=args.neighbor_size, mu=args.mu)
S3_mirna = snf.compute.affinity_matrix(dist_mirna, K=args.neighbor_size, mu=args.mu)
S4_rnaseq = snf.compute.affinity_matrix(dist_rnaseq, K=args.neighbor_size, mu=args.mu)
S5_rppa = snf.compute.affinity_matrix(dist_rppa, K=args.neighbor_size, mu=args.mu)
print("finish building individual similarity network!")

# Do SNF2 diffusion
S1_df = pd.DataFrame(data=S1_cnv, index=original_order[0], columns=original_order[0])
S2_df = pd.DataFrame(data=S2_meth, index=original_order[1], columns=original_order[1])
S3_df = pd.DataFrame(data=S3_mirna, index=original_order[2], columns=original_order[2])
S4_df = pd.DataFrame(data=S4_rnaseq, index=original_order[3], columns=original_order[3])
S5_df = pd.DataFrame(data=S5_rppa, index=original_order[4], columns=original_order[4])

fused_networks = snf2(
    args,
    [S1_df, S2_df, S3_df, S4_df, S5_df],
    dicts_common=dicts_common,
    dicts_unique=dicts_unique,
    original_order=original_order,
)

S1_fused = fused_networks[0]
S2_fused = fused_networks[1]
S3_fused = fused_networks[2]
S4_fused = fused_networks[3]
S5_fused = fused_networks[4]

# t-sne extraction
result_dir = os.path.join(d, "results/SKCM")
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
"""
w1_tsne = tsne_p(S1_fused.values, no_dims=50)
np.savetxt(os.path.join(result_dir, "tsne_embedding/w1_tsne.csv"), w1_tsne, delimiter=",")

w2_tsne = tsne_p(S2_fused.values, no_dims=50)
np.savetxt(os.path.join(result_dir, "tsne_embedding/w2_tsne.csv"), w2_tsne, delimiter=",")

w3_tsne = tsne_p(S3_fused.values, no_dims=50)
np.savetxt(os.path.join(result_dir, "tsne_embedding/w3_tsne.csv"), w3_tsne, delimiter=",")

w4_tsne = tsne_p(S4_fused.values, no_dims=50)
np.savetxt(os.path.join(result_dir, "tsne_embedding/w4_tsne.csv"), w4_tsne, delimiter=",")

w5_tsne = tsne_p(S5_fused.values, no_dims=50)
np.savetxt(os.path.join(result_dir, "tsne_embedding/w5_tsne.csv"), w5_tsne, delimiter=",")


w1_tsne = np.loadtxt(
    os.path.join(result_dir, "tsne_embedding/w1_tsne.csv"), delimiter=","
)
w2_tsne = np.loadtxt(
    os.path.join(result_dir, "tsne_embedding/w2_tsne.csv"), delimiter=","
)
w3_tsne = np.loadtxt(
    os.path.join(result_dir, "tsne_embedding/w3_tsne.csv"), delimiter=","
)
w4_tsne = np.loadtxt(
    os.path.join(result_dir, "tsne_embedding/w4_tsne.csv"), delimiter=","
)
w5_tsne = np.loadtxt(
    os.path.join(result_dir, "tsne_embedding/w5_tsne.csv"), delimiter=","
)

# matching
embed_w1 = pd.DataFrame(data=w1_tsne, index=original_order[0])
embed_w2 = pd.DataFrame(data=w2_tsne, index=original_order[1])
embed_w3 = pd.DataFrame(data=w3_tsne, index=original_order[2])
embed_w4 = pd.DataFrame(data=w4_tsne, index=original_order[3])
embed_w5 = pd.DataFrame(data=w5_tsne, index=original_order[4])

S_final = kernel_matching(
    [embed_w1, embed_w2, embed_w3, embed_w4, embed_w5],
    dicts_common=dicts_common,
    dicts_unique=dicts_unique,
    alpha=0.1,
    matching_iter=49,
)
"""

S_final = tsne_p_deep(
    args,
    dicts_commonIndex,
    dict_sampleToIndexs,
    [cnv.values, meth.values, mirna.values, rnaseq.values, rppa.values],
    [
        S1_fused.values,
        S2_fused.values,
        S3_fused.values,
        S4_fused.values,
        S5_fused.values,
    ],
)

#dist_final = dist2(S_final.values, S_final.values)
dist_final = dist2(S_final, S_final)
Wall_final = snf.compute.affinity_matrix(dist_final, K=20, mu=0.5)

best, second = snf.get_n_clusters(Wall_final)
print(best, second)
labels = spectral_clustering(Wall_final, n_clusters=3)

# TSNE plots
from sklearn.manifold import TSNE

# X_embedded = TSNE(n_components=2, metric='precomputed').fit_transform(1-Wall_final.values)
X_embedded = TSNE(n_components=2).fit_transform(S_final)
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, s=1.5, cmap="Spectral")
plt.title("t-SNE visualization of union SKCM patients")
save_path = os.path.join(result_dir, "tSNE.png")
plt.savefig(save_path)
print("Save visualization at {}".format(save_path))

# save result
S_final_df = pd.DataFrame(data=S_final, index=dict_sampleToIndexs.keys())
S_final_df["spectral"] = labels

survival_ = os.path.join(testdata_dir, "SKCM_survival.csv")
survival = pd.read_csv(survival_, index_col=0)
survival.rename(
    {"Overall Survival (Months)": "timetoevent", "Overall Survival Status": "event"},
    axis=1,
    inplace=True,
)

S_final_df = S_final_df.merge(survival, how="inner", left_index=True, right_index=True)

S_final_df.to_csv(os.path.join(result_dir, "allEmbedding.csv"))