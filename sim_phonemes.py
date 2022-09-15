from entropy_of_phonemes import *
import numpy as np
from scipy.spatial import distance
from matplotlib import cm as cm
import scipy.cluster.hierarchy as shc
from sklearn.manifold import MDS
from scipy.spatial.distance import squareform

with open("saved_dictionary_codebook_1.pkl", "rb") as f:
    phn_dict = pickle.load(f)
set_of_val = set(list(flatten(phn_dict.values())))
abs_discount = 0.00000000002


def absolute_discounting(dist: list):
    dist = np.array(dist)
    dist_ret = dist
    ind_nonzero = tuple(np.nonzero(dist))
    ind_zero = np.where((dist == 0))[0]
    dist_ret[ind_nonzero] = list(
        ((dist[ind_nonzero] - abs_discount) / sum(dist))
        + ((abs_discount * len(ind_nonzero)) / sum(dist)) * (1 / len(set_of_val))
    )
    dist_ret[ind_zero] = ((abs_discount * len(ind_nonzero)) / sum(dist)) * (1 / len(set_of_val))
    
    return dist_ret
#creating distribution mapping for each phoneme
phn_to_dist = {}
for phn_name in phn_dict.keys():
    temp_lst = adding_missing_phn_count(set_of_val, phn_dict, phn_name)
    temp_lst_smoothed = absolute_discounting(temp_lst)
    phn_to_dist[phn_name] = temp_lst_smoothed

#calculating similarity
sim_mt = np.zeros((len(phn_to_dist), len(phn_to_dist)))
for num1, i in enumerate(phn_to_dist.keys()):
    for num2, j in enumerate(phn_to_dist.keys()):
        sim_mt[num1][num2] = distance.jensenshannon(phn_to_dist[i], phn_to_dist[j])
        print(f"sim between {i} and {j}:", sim_mt[num1][num2])

#plotting similarity
cmap = cm.get_cmap("YlGnBu")
fig, ax = plt.subplots(figsize=(7, 7))
cax = ax.matshow(1.0-sim_mt, interpolation="nearest", cmap=cmap)
# ax.grid(True)
plt.title("Similarity matrix")
plt.xticks(range(len(phn_to_dist)), phn_to_dist.keys(), rotation=90)
plt.yticks(range(len(phn_to_dist)), phn_to_dist.keys())
fig.colorbar(cax, ticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.savefig("sim.pdf", bbox_inches="tight")
plt.figure()

# plt.subplots(figsize=(5, 5))
selected_data = sim_mt
clusters = shc.linkage(squareform(selected_data), method="ward", metric="euclidean")
shc.dendrogram(Z=clusters, labels=list(phn_to_dist.keys()), orientation="left")
plt.savefig("lang.pdf", bbox_inches="tight")
plt.figure()

# MDS plot

mds_2d = MDS(n_components=2, dissimilarity='precomputed', n_jobs=-1 ).fit_transform(sim_mt)
sns.scatterplot(mds_2d[:,0], mds_2d[:,1])
for i, phn in enumerate (phn_to_dist.keys()):
    plt.annotate(phn, (mds_2d[:,0][i], mds_2d[:,1][i]) )
plt.xlabel('MDS axis 1')
plt.ylabel('MDS axis 2')
plt.show()