import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
import math
import pickle
import numpy as np
plt.figure(figsize=(10, 7))
plt.title("Customers Dendrogram")
config = ['gu_in','kn_in','ml_in_t','mr_in','ne_np','sd_in','ur_pk','as_in','bn_in','hi_in','or_in','pa_in','ta_in','te_in']
config_names = ['gujrati','kannada','malayalam','marathi','nepali','sindhi','urdu','assamese','bengali','hindi','oriya','punjabi','tamil','telugu']
sim_mt = np.zeros((len(config),len(config)))
counter_lang = {}
def cosine_dic(dic1,dic2):
    numerator = 0
    dena = 0
    for key1,val1 in dic1.items():
        numerator += val1*dic2.get(key1,0.0)
        dena += val1*val1
    denb = 0
    for val2 in dic2.values():
        denb += val2*val2
    return numerator/math.sqrt(dena*denb)

for i in config:
    with open( f'{i}.pkl', 'rb+') as f:
        counter_lang[i] = pickle.load(f)
for num1,j in enumerate(counter_lang.keys()):
    for num2,k in enumerate(counter_lang.keys()):
        print(f'sim between {j} and {k}:',cosine_dic(counter_lang[j],counter_lang[k]))
        sim_mt[num1][num2] = cosine_dic(counter_lang[j],counter_lang[k])

# Selecting Annual Income and Spending Scores by index
selected_data = sim_mt
clusters = shc.linkage(selected_data, 
            method='ward', 
            metric="euclidean")
shc.dendrogram(Z=clusters,labels = config_names)
plt.show()