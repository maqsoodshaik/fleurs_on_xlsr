import os
import pickle

from sim_phonemes import similarity_calculation,phn_dict_generator
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import timit_to_ipa
def main():
    codebook = 1
    folder_name = "timit_pkl"
    pickle_path = (
        f"/Users/mohammedmaqsoodshaik/Desktop/hiwi/task1/{folder_name}/codebook{codebook}/"
    )
    rootdir = "/Users/mohammedmaqsoodshaik/Desktop/hiwi/task1/TIMIT/timit/"

    

    phn_dict_m = {}
    starts_with = "m"
    phn_dict_m = phn_dict_generator(phn_dict_m,pickle_path,rootdir, starts_with)
    phn_dict_f = {}
    starts_with = "f"
    phn_dict_f = phn_dict_generator(phn_dict_f,pickle_path,rootdir, starts_with)
    abs_discount = 0.00000000002
    sim_mt = similarity_calculation(phn_dict_m, phn_dict_f,abs_discount)
    sim_mt = np.array(sim_mt)
    sim_mt_diag = sim_mt.diagonal()
    sim_mt_diag = 1.0 - sim_mt_diag
    labels = [timit_to_ipa.timit_2_ipa[k] for k in phn_dict_m.keys()]
    sorted_phn_sim = [x for _, x in sorted(zip(sim_mt_diag, labels))]
    g = sns.barplot(sorted_phn_sim, sorted(sim_mt_diag))
    g.set_xticklabels(g.get_xticklabels(), rotation=-90)
    plt.xlabel("phonemes")
    plt.ylabel("similarity")
    plt.show()

if __name__ == "__main__":
    main()