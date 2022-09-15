import operator
from collections import OrderedDict
from scipy.stats import entropy
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from collections import Counter
def flatten(l): #flattening list of lists
    return [item for sublist in l for item in sublist]
def adding_missing_phn_count(set_of_val,phn_dict,phn_name):
    phn_map_counts = dict(Counter(phn_dict[phn_name]))
    ##sorting counter dictionary
    # sorted_tuples = sorted(dict1.items(), key=operator.itemgetter(1))
    # # print(sorted_tuples)  # [(1, 1), (3, 4), (2, 9)]

    # sorted_dict = OrderedDict()
    # for k, v in sorted_tuples:

    #     sorted_dict[k] = v
    temp_lst = []
    #creating list of counts of code entries even including the codeentries which are not present
    for val in set_of_val:
        if val in phn_map_counts:
            temp_lst.append(phn_map_counts[val])
        else:
            temp_lst.append(0)
    temp_lst = [float(i)/sum(temp_lst) for i in temp_lst]
    return temp_lst
if __name__ == "__main__":
    with open("saved_dictionary_codebook_1.pkl", "rb") as f:
        phn_dict = pickle.load(f)


    #set of codebook entries
    set_of_val = set(list(flatten(phn_dict.values())))
    entropy_lst = []

    for phn_name in phn_dict.keys():
        temp_lst = adding_missing_phn_count(set_of_val,phn_dict,phn_name)
        
        #entropy calculation
        print(f'{phn_name}:{entropy(temp_lst)}')
        entropy_lst.append(entropy(temp_lst))
        # print(f"{dict_val}-{dict(sorted_dict).popitem()[0]}")
    #plotting the entropy
    phn_dict_sorted_based_on_entropy = [x for _,x in sorted(zip(entropy_lst,list(phn_dict.keys())))]#sorting phonemes dictionary based on the corresponding entropy
    g = sns.barplot(phn_dict_sorted_based_on_entropy, sorted(entropy_lst))
    g.set_xticklabels(g.get_xticklabels(), rotation=-90)
    plt.xlabel('phonemes')
    plt.ylabel('entropy')
    plt.show()
    # ax.set_title('Entropy vs phonemes')
#entropy for 55 elements- 4.007333185232471
#entropy for 100 elements - 4.605170185988092
