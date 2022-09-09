import torch
import pickle
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm as cm
from scipy import stats
from scipy.stats import norm
import numpy as np

cmap = cm.get_cmap("YlGnBu")
# def cosine_dic(dic1,dic2):
#     numerator = 0
#     dena = 0
#     for key1,val1 in dic1.items():
#         numerator += val1*dic2.get(key1,0.0)
#         dena += val1*val1
#     denb = 0
#     for val2 in dic2.values():
#         denb += val2*val2
#     return numerator/math.sqrt(dena*denb)


# counter_lang = {}
config = ["as_in", "bn_in", "hi_in", "or_in", "pa_in", "ta_in", "te_in"]
config = [
    "gu_in",
    "kn_in",
    "ml_in_t",
    "mr_in",
    "ne_np",
    "sd_in",
    "ur_pk",
    "as_in",
    "bn_in",
    "hi_in",
    "or_in",
    "pa_in",
    "ta_in",
    "te_in",
]
config_names = [
    "gujrati",
    "kannada",
    "malayalam",
    "marathi",
    "nepali",
    "sindhi",
    "urdu",
    "assamese",
    "bengali",
    "hindi",
    "oriya",
    "punjabi",
    "tamil",
    "telugu",
]


# extract =0
# if extract ==1:
#     for i in config:
#         counter = {}
#         proj = torch.load(i+'.pt')
#         for letter in proj:
#             letter = str(letter)
#             if letter not in counter:
#                 counter[letter] = 0
#             counter[letter] += 1
#         counter = {k: v /proj.shape[0]  for k, v in counter.items()}
#         with open( f'{i}.pkl', 'wb+') as f:
#             pickle.dump(counter, f)
#         print(max(counter.values()))
#         counter_lang[i] = counter

# else:
#     for i in config:
#         with open( f'{i}.pkl', 'rb+') as f:
#             counter_lang[i] = pickle.load(f)
#     for num1,j in enumerate(counter_lang.keys()):
#         for num2,k in enumerate(counter_lang.keys()):
#             print(f'sim between {j} and {k}:',cosine_dic(counter_lang[j],counter_lang[k]))
#             sim_mt[num1][num2] = cosine_dic(counter_lang[j],counter_lang[k])


# function to compute KL Divergence
"""KL Divergence(P|Q)"""


def KLD(p_probs, q_probs):
    KLD = p_probs * np.log(p_probs / q_probs)
    return np.sum(KLD)


# function to compute JS Divergence
def JSD(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    # normalize
    p /= p.sum()
    q /= q.sum()
    m = (p + q) / 2
    return (KLD(p, m) + KLD(q, m)) / 2


sim_mt = np.zeros((len(config), len(config)))
# create the data distribution
for num1, i in enumerate(config):
    for num2, j in enumerate(config):
        proj = torch.load(i + ".pt")
        proj2 = torch.load(j + ".pt")
        sim_mt[num1][num2] = JSD(proj, proj2)
        print(f"sim between {i} and {j}:", sim_mt[num1][num2])


fig, ax = plt.subplots(figsize=(7, 7))
cax = ax.matshow(sim_mt, interpolation="nearest", cmap=cmap)
# ax.grid(True)
plt.title("Similarity matrix")
plt.xticks(range(len(config)), config_names, rotation=90)
plt.yticks(range(len(config)), config_names)
fig.colorbar(
    cax, ticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.90, 0.95, 1]
)
plt.show()
