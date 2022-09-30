from entropy_of_phonemes import *
import numpy as np
from scipy.spatial import distance
from matplotlib import cm as cm
import scipy.cluster.hierarchy as shc
from sklearn.manifold import MDS
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import random
from timit_feature_extraction import codes_low_high,phn_ind
import os
import timit_to_ipa
# from  male_female_timit_analysis import phn_dict_generator



def absolute_discounting(dist: list,abs_discount,set_of_val):
    dist = np.array(dist)
    dist_ret = dist
    ind_nonzero = tuple(np.nonzero(dist))
    ind_zero = np.where((dist == 0))[0]
    dist_ret[ind_nonzero] = list(
        ((dist[ind_nonzero] - abs_discount) / sum(dist))
        + ((abs_discount * len(ind_nonzero)) / sum(dist)) * (1 / len(set_of_val))
    )
    dist_ret[ind_zero] = ((abs_discount * len(ind_nonzero)) / sum(dist)) * (
        1 / len(set_of_val)
    )

    return dist_ret


# creating distribution mapping for each phoneme
def smoothing_dist(phn_dict, set_of_val,abs_discount):
    phn_to_dist = {}
    for phn_name in phn_dict:
        temp_lst = adding_missing_phn_count(set_of_val, phn_dict, phn_name)
        temp_lst_smoothed = absolute_discounting(temp_lst,abs_discount,set_of_val)
        phn_to_dist[phn_name] = temp_lst_smoothed
    return phn_to_dist


# calculating similarity
def similarity_calculation(phn_to_dist_1, phn_to_dist_2,abs_discount):
    set_of_val = set(
        list(flatten(phn_to_dist_1.values())) + list(flatten(phn_to_dist_2.values()))
    )
    phn_to_dist_1 = smoothing_dist(phn_to_dist_1, set_of_val,abs_discount)
    phn_to_dist_2 = smoothing_dist(phn_to_dist_2, set_of_val,abs_discount)
    sim_mt = np.zeros((len(phn_to_dist_1), len(phn_to_dist_2)))
    for num1, i in enumerate(phn_to_dist_1.keys()):
        for num2, j in enumerate(phn_to_dist_2.keys()):
            sim_mt[num1][num2] = distance.jensenshannon(
                phn_to_dist_1[i], phn_to_dist_2[j]
            )
            print(f"sim between {i} and {j}:", sim_mt[num1][num2])
    return sim_mt


# plotting similarity
def plot_sim(sim_mt, phn_to_dist_1_keys, phn_to_dist_2_keys):
    cmap = cm.get_cmap("YlGnBu")
    fig, ax = plt.subplots(figsize=(7, 7))
    cax = ax.matshow(1.0 - sim_mt, interpolation="nearest", cmap=cmap)
    # ax.grid(True)
    plt.title("Similarity matrix")
    plt.xticks(range(sim_mt.shape[0]), phn_to_dist_1_keys, rotation=90,fontsize = 10)
    plt.yticks(range(sim_mt.shape[1]), phn_to_dist_1_keys,fontsize =10)
    fig.colorbar(cax, ticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.savefig("sim_phn.pdf", bbox_inches="tight")
    plt.figure()
    selected_data = sim_mt
    clusters = shc.linkage(squareform(selected_data), method="ward", metric="euclidean",optimal_ordering = True)
    shc.dendrogram(Z=clusters, labels=list(phn_to_dist_1_keys), orientation="left",distance_sort = True)
    plt.savefig("dend_phn.pdf", bbox_inches="tight")


# MDS plot
def mds_plot(sim_mt, phn_to_dist_1_keys):
    # mds_2d = MDS(n_components=2, dissimilarity="precomputed", n_jobs=-1).fit_transform(
    #     sim_mt
    # )
    mds_2d = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=3,random_state = 0).fit_transform(sim_mt)
    # kmeans = KMeans(n_clusters=7, random_state=0).fit(mds_2d)
    # colors_cluster = kmeans.labels_
    sns.scatterplot(mds_2d[:, 0], mds_2d[:, 1])#,c = colors_cluster
    for i, phn in enumerate(phn_to_dist_1_keys):
        v = random.uniform(0, 0.5)
        plt.annotate(phn, (mds_2d[:, 0][i], mds_2d[:, 1][i]))
    plt.xlabel("TSNE axis 1")
    plt.ylabel("TSNE axis 2")
def phn_dict_generator(phn_dict,pickle_path,rootdir, starts_with = ""):
        for subdir, dirs, files in os.walk(pickle_path):
            for file in files:
                if subdir.split("/")[-1].startswith(starts_with) and "pkl" in file:
                    with open(os.path.join(subdir, file), "rb") as f:
                        output = pickle.load(f)
                    low_lst, high_lst = codes_low_high(
                        os.path.join(
                            rootdir + "/".join(subdir.split("/")[-3:]),
                            file.replace("pkl", "wav"),
                        ),
                        output,
                    )
                    phnpath = os.path.join(
                        rootdir + "/".join(subdir.split("/")[-3:]),
                        file.replace("pkl", "phn"),
                    )
                    phn_dict = phn_ind(phnpath, low_lst, high_lst, output, phn_dict)
        return phn_dict

def main():
    # with open("saved_dictionary.pkl", "rb") as f:
    #     phn_dict_o = pickle.load(f)
    phn_dict ={}
    codebook = 2
    folder_name = "timit_pkl"
    pickle_path = (
        f"/Users/mohammedmaqsoodshaik/Desktop/hiwi/task1/{folder_name}/codebook{codebook}/"
    )
    rootdir = "/Users/mohammedmaqsoodshaik/Desktop/hiwi/task1/TIMIT/timit/"
    phn_dict = phn_dict_generator(phn_dict,pickle_path,rootdir)
    # set_of_val = set(list(flatten(phn_dict.values())))
    # print(f'set:{len(set_of_val)}')
    abs_discount = 0.00000000002
    sim_mt = similarity_calculation(phn_dict, phn_dict,abs_discount)
    labels = [timit_to_ipa.timit_2_ipa[k] for k in phn_dict.keys()]
    plot_sim(sim_mt, labels, labels)
    # plt.figure()
    # mds_plot(sim_mt, labels)
    plt.show()

if __name__ == "__main__":
    main()
