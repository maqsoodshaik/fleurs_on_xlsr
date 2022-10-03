import operator
from collections import OrderedDict
from scipy.stats import entropy
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import timit_to_ipa
from sim_phonemes import phn_dict_generator,flatten,adding_missing_phn_count


def main():
    #load from a single file
    # with open("saved_dictionary_codebook_1.pkl", "rb") as f:
    #     phn_dict = pickle.load(f)
    #load from multiple file
    phn_dict1 ={}
    codebook = 1
    folder_name = "timit_pkl_only_english"
    model = "wav2vec2.0"#"xlsr-53"
    pickle_path = (
        f"/Users/mohammedmaqsoodshaik/Desktop/hiwi/task1/{folder_name}/codebook{codebook}/"
    )
    rootdir = "/Users/mohammedmaqsoodshaik/Desktop/hiwi/task1/TIMIT/timit/"
    phn_dict1 = phn_dict_generator(phn_dict1,pickle_path,rootdir)
    phn_dict2 ={}
    codebook = 2
    folder_name = "timit_pkl_only_english"
    pickle_path = (
        f"/Users/mohammedmaqsoodshaik/Desktop/hiwi/task1/{folder_name}/codebook{codebook}/"
    )
    rootdir = "/Users/mohammedmaqsoodshaik/Desktop/hiwi/task1/TIMIT/timit/"
    phn_dict2 = phn_dict_generator(phn_dict2,pickle_path,rootdir)
    phn_dict = {}
    def append_320(dict2values):
        return list(map(lambda x:x+320, dict2values))
    for k,v in phn_dict1.items():
        phn_dict[k]=v+append_320(phn_dict2[k])
    # with open("saved_dictionary.pkl", "rb") as f:
    #     phn_dict_2 = pickle.load(f)
    # from collections import defaultdict
    # dates_dict = defaultdict(list)
    # for key, date in phn_dict.items():
    #     dates_dict[key]+=date
    # for key, date in phn_dict_2.items():
    #     dates_dict[key]+=date
    # phn_dict = dates_dict
    #set of codebook entries
    set_of_val = set(list(flatten(phn_dict.values())))
    entropy_lst = []
    print(f"default entropy:{entropy([1/len(set_of_val)]*len(set_of_val))}")
    for phn_name in phn_dict:
        temp_lst = adding_missing_phn_count(set_of_val,phn_dict,phn_name)
        
        #entropy calculation
        print(f'{phn_name}:{entropy(temp_lst)}')
        entropy_lst.append(entropy(temp_lst))
        # print(f"{dict_val}-{dict(sorted_dict).popitem()[0]}")
    #plotting the entropy
    labels = [timit_to_ipa.timit_2_ipa[k] for k in phn_dict.keys()]
    phn_dict_sorted_based_on_entropy = [x for _,x in sorted(zip(entropy_lst,labels),reverse=True)]#sorting phonemes dictionary based on the corresponding entropy
    g = sns.barplot(phn_dict_sorted_based_on_entropy, sorted(entropy_lst,reverse=True))
    g.set_xticklabels(g.get_xticklabels(), rotation=-90)
    plt.xlabel('phonemes',fontsize = 0.1)
    plt.ylabel('entropy')
    plt.title(f'Entropy of each TIMIT dataset phoneme according to codebook entries of {model} model')
    # plt.savefig("Entropy_2.pdf", bbox_inches="tight")
    plt.show()
if __name__ == "__main__":
   main()
#entropy for 55 elements- 4.007333185232471
#entropy for 100 elements - 4.605170185988092
