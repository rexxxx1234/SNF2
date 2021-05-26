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
testdata_CCLE = "/scratch/gobi2/rexma/snf2_cellline/CCLE/Breast"
testdata_gCSI = "/scratch/gobi2/rexma/snf2_cellline/gCSI/Breast"
testdata_GDSC = "/scratch/gobi2/rexma/snf2_cellline/GDSC/Breast"

cnv_CCLE = os.path.join(testdata_CCLE, "cnv_57x22768.csv")
cnv_gCSI = os.path.join(testdata_gCSI, "cnv_28x26168.csv")
cnv_GDSC = os.path.join(testdata_GDSC, "cnv_46x22738.csv")

rna_CCLE = os.path.join(testdata_CCLE, "rna_59x20024.csv")
rna_gCSI = os.path.join(testdata_gCSI, "rna_66x60662.csv")
rna_GDSC = os.path.join(testdata_GDSC, "rna_39x11894.csv")

mut_CCLE = os.path.join(testdata_CCLE, "mut_53x1667.csv")
mut_gCSI = os.path.join(testdata_gCSI, "mut_23x45.csv")
mut_GDSC = os.path.join(testdata_GDSC, "mut_51x278.csv")

cnv1 = pd.read_csv(cnv_CCLE, index_col=0)
cnv2 = pd.read_csv(cnv_gCSI, index_col=0)
cnv3 = pd.read_csv(cnv_GDSC, index_col=0)

rna1 = pd.read_csv(rna_CCLE, index_col=0)
rna2 = pd.read_csv(rna_gCSI, index_col=0)
rna3 = pd.read_csv(rna_GDSC, index_col=0)

mut1 = pd.read_csv(mut_CCLE, index_col=0)
mut2 = pd.read_csv(mut_gCSI, index_col=0)
mut3 = pd.read_csv(mut_GDSC, index_col=0)

print("finish loading data!")

########################################################
'''
    First we integrate modality from different datasets
'''
#########################################################

# intergrate cnv
print("start integrating cnv data!")
(
    dicts_common,
    dicts_commonIndex,
    dict_sampleToIndexs,
    dicts_unique,
    original_order,
) = data_indexing([cnv1, cnv2, cnv3])

dist_cnv1 = dist2(cnv1.values, cnv1.values)
dist_cnv2 = dist2(cnv2.values, cnv2.values)
dist_cnv3 = dist2(cnv3.values, cnv3.values)

S1_cnv1 = snf.compute.affinity_matrix(dist_cnv1, K=args.neighbor_size, mu=args.mu)
S1_cnv2 = snf.compute.affinity_matrix(dist_cnv2, K=args.neighbor_size, mu=args.mu)
S1_cnv3 = snf.compute.affinity_matrix(dist_cnv3, K=args.neighbor_size, mu=args.mu)

S1_df = pd.DataFrame(data=S1_cnv1, index=original_order[0], columns=original_order[0])
S2_df = pd.DataFrame(data=S1_cnv2, index=original_order[1], columns=original_order[1])
S3_df = pd.DataFrame(data=S1_cnv3, index=original_order[2], columns=original_order[2])

fused_networks = snf2(
    args,
    [S1_df, S2_df, S3_df],
    dicts_common=dicts_common,
    dicts_unique=dicts_unique,
    original_order=original_order,
)

S1_fused = fused_networks[0]
S2_fused = fused_networks[1]
S3_fused = fused_networks[2]

S_final_cnv = tsne_p_deep(
    args,
    dicts_commonIndex,
    dict_sampleToIndexs,
    [
        S1_fused.values,
        S2_fused.values,
        S3_fused.values,
    ],
)
S_final_cnvdf = pd.DataFrame(data=S_final_cnv, index=dict_sampleToIndexs.keys())
print("finish integrating cnv data!")
print("cnv shape: ", S_final_cnvdf.shape)

# intergrate rna
print("start integrating rna data!")
(
    dicts_common,
    dicts_commonIndex,
    dict_sampleToIndexs,
    dicts_unique,
    original_order,
) = data_indexing([rna1, rna2, rna3])

dist_rna1 = dist2(rna1.values, rna1.values)
dist_rna2 = dist2(rna2.values, rna2.values)
dist_rna3 = dist2(rna3.values, rna3.values)

S1_rna1 = snf.compute.affinity_matrix(dist_rna1, K=args.neighbor_size, mu=args.mu)
S1_rna2 = snf.compute.affinity_matrix(dist_rna2, K=args.neighbor_size, mu=args.mu)
S1_rna3 = snf.compute.affinity_matrix(dist_rna3, K=args.neighbor_size, mu=args.mu)

S1_df = pd.DataFrame(data=S1_rna1, index=original_order[0], columns=original_order[0])
S2_df = pd.DataFrame(data=S1_rna2, index=original_order[1], columns=original_order[1])
S3_df = pd.DataFrame(data=S1_rna3, index=original_order[2], columns=original_order[2])

fused_networks = snf2(
    args,
    [S1_df, S2_df, S3_df],
    dicts_common=dicts_common,
    dicts_unique=dicts_unique,
    original_order=original_order,
)

S1_fused = fused_networks[0]
S2_fused = fused_networks[1]
S3_fused = fused_networks[2]

S_final_rna = tsne_p_deep(
    args,
    dicts_commonIndex,
    dict_sampleToIndexs,
    [
        S1_fused.values,
        S2_fused.values,
        S3_fused.values,
    ],
)
S_final_rnadf = pd.DataFrame(data=S_final_rna, index=dict_sampleToIndexs.keys())
print("finish integrating rna data!")
print("rna shape: ", S_final_rnadf.shape)

# intergrate mut
print("start integrating mut data!")
(
    dicts_common,
    dicts_commonIndex,
    dict_sampleToIndexs,
    dicts_unique,
    original_order,
) = data_indexing([mut1, mut2, mut3])
print("finish indexing data!")

dist_mut1 = dist2(mut1.values, mut1.values)
dist_mut2 = dist2(mut2.values, mut2.values)
dist_mut3 = dist2(mut3.values, mut3.values)

S1_mut1 = snf.compute.affinity_matrix(dist_mut1, K=args.neighbor_size, mu=args.mu)
S1_mut2 = snf.compute.affinity_matrix(dist_mut2, K=args.neighbor_size, mu=args.mu)
S1_mut3 = snf.compute.affinity_matrix(dist_mut3, K=args.neighbor_size, mu=args.mu)

S1_df = pd.DataFrame(data=S1_mut1, index=original_order[0], columns=original_order[0])
S2_df = pd.DataFrame(data=S1_mut2, index=original_order[1], columns=original_order[1])
S3_df = pd.DataFrame(data=S1_mut3, index=original_order[2], columns=original_order[2])

fused_networks = snf2(
    args,
    [S1_df, S2_df, S3_df],
    dicts_common=dicts_common,
    dicts_unique=dicts_unique,
    original_order=original_order,
)

S1_fused = fused_networks[0]
S2_fused = fused_networks[1]
S3_fused = fused_networks[2]

S_final_mut = tsne_p_deep(
    args,
    dicts_commonIndex,
    dict_sampleToIndexs,
    [
        S1_fused.values,
        S2_fused.values,
        S3_fused.values,
    ],
)
S_final_mutdf = pd.DataFrame(data=S_final_mut, index=dict_sampleToIndexs.keys())
print("finish integrating mut data!")
print("mut shape: ", S_final_mutdf.shape)

########################################################
'''
    Next, we are going to integrate diffent integrated modality
'''
#########################################################

# data indexing
(
    dicts_common,
    dicts_commonIndex,
    dict_sampleToIndexs,
    dicts_unique,
    original_order,
) = data_indexing([S_final_cnvdf, S_final_rnadf, S_final_mutdf])
print("finish indexing data!")


dist_cnv = dist2(S_final_cnvdf.values, S_final_cnvdf.values)
dist_rna = dist2(S_final_rnadf.values, S_final_rnadf.values)
dist_mut = dist2(S_final_mutdf.values, S_final_mutdf.values)

S1_cnv = snf.compute.affinity_matrix(dist_cnv, K=args.neighbor_size, mu=args.mu)
S2_rna = snf.compute.affinity_matrix(dist_rna, K=args.neighbor_size, mu=args.mu)
S3_mut = snf.compute.affinity_matrix(dist_mut, K=args.neighbor_size, mu=args.mu)

# Do SNF2 diffusion
S1_df = pd.DataFrame(data=S1_cnv, index=original_order[0], columns=original_order[0])
S2_df = pd.DataFrame(data=S2_rna, index=original_order[1], columns=original_order[1])
S3_df = pd.DataFrame(data=S3_mut, index=original_order[2], columns=original_order[2])

fused_networks = snf2(
    args,
    [S1_df, S2_df, S3_df],
    dicts_common=dicts_common,
    dicts_unique=dicts_unique,
    original_order=original_order,
)

S1_fused = fused_networks[0]
S2_fused = fused_networks[1]
S3_fused = fused_networks[2]

result_dir = os.path.join(d, "results/cellline/Breast")
if not os.path.exists(result_dir):
    os.makedirs(result_dir)


S_final = tsne_p_deep(
    args,
    dicts_commonIndex,
    dict_sampleToIndexs,
    [
        S1_fused.values,
        S2_fused.values,
        S3_fused.values,
    ],
)

# dist_final = dist2(S_final.values, S_final.values)
dist_final = dist2(S_final, S_final)
Wall_final = snf.compute.affinity_matrix(dist_final, K=args.neighbor_size, mu=args.mu)

best, second = snf.get_n_clusters(Wall_final)
print(best, second)
labels = spectral_clustering(Wall_final, n_clusters=best)

# TSNE plots
from sklearn.manifold import TSNE

# X_embedded = TSNE(n_components=2, metric='precomputed').fit_transform(1-Wall_final.values)
X_embedded = TSNE(n_components=2).fit_transform(S_final)
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, s=1.5, cmap="Spectral")
plt.title("t-SNE visualization of union breast cellline")
save_path = os.path.join(result_dir, "tSNE.png")
plt.savefig(save_path)
print("Save visualization at {}".format(save_path))