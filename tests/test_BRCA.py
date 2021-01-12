import numpy as np
import pandas as pd
import snf
from sklearn.cluster import spectral_clustering
from sklearn.metrics import v_measure_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import sys
import os
d = os.path.dirname(os.getcwd())
sys.path.insert(0, d)
from snf2.embedding import tsne_p
from snf2.main import dist2, snf2, kernel_matching
from snf2.util import data_indexing

# read the data
testdata_dir = os.path.join(d, "data/snf2_cancers/BRCA")
cnv_ = os.path.join(testdata_dir, "cnv_1080x24776.csv")
meth_ = os.path.join(testdata_dir, "meth_784x16474.csv")
mirna_ = os.path.join(testdata_dir, "mirna_756x1046.csv")
rnaseq_ = os.path.join(testdata_dir, "rnaseq_1093x20531.csv")
rppa_ = os.path.join(testdata_dir, "rppa_887x222.csv.csv")


cnv = pd.read_csv(cnv_, index_col=0)
meth = pd.read_csv(meth_, index_col=0)
mirna = pd.read_csv(mirna_, index_col=0)
rnaseq = pd.read_csv(rnaseq_, index_col=0)
rppa = pd.read_csv(rppa_, index_col=0)

# data indexing
dicts_common, dicts_unique, original_order = data_indexing([cnv, meth, mirna, rnaseq, rppa])


# build similarity networks for each motality
dist_cnv = dist2(cnv.values, cnv.values)
dist_meth = dist2(meth.values, meth.values)
dist_mirna = dist2(mirna.values, mirna.values)
dist_rnaseq = dist2(rnaseq.values, rnaseq.values)
dist_rppa = dist2(rppa.values, rppa.values)

S1_cnv = snf.compute.affinity_matrix(dist_cnv, K=20, mu=0.5)
S2_meth = snf.compute.affinity_matrix(dist_meth, K=20, mu=0.5)
S3_mirna = snf.compute.affinity_matrix(dist_mirna, K=20, mu=0.5)
S4_rnaseq = snf.compute.affinity_matrix(dist_rnaseq, K=20, mu=0.5)
S5_rppa = snf.compute.affinity_matrix(dist_rppa, K=20, mu=0.5)

# Do SNF2 diffusion
S1_df = pd.DataFrame(data=S1_cnv, index=original_order[0], columns=original_order[0])
S2_df = pd.DataFrame(data=S2_meth, index=original_order[1], columns=original_order[1])
S3_df = pd.DataFrame(data=S3_mirna, index=original_order[2], columns=original_order[2])
S4_df = pd.DataFrame(data=S4_rnaseq, index=original_order[3], columns=original_order[3])
S5_df = pd.DataFrame(data=S5_rppa, index=original_order[4], columns=original_order[4])

fused_networks = snf2([S1_df, S2_df, S3_df, S4_df, S5_df], 
                      dicts_common=dicts_common,
                      dicts_unique=dicts_unique,
                      original_order=original_order)

S1_fused = fused_networks[0]
S2_fused = fused_networks[1]
S3_fused = fused_networks[2]
S4_fused = fused_networks[3]
S5_fused = fused_networks[4]
