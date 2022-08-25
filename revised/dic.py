import torch
import pickle
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm as cm
from scipy import stats
from scipy.stats import norm
import numpy as np
from scipy.spatial import distance
import scipy.cluster.hierarchy as shc
cmap = cm.get_cmap('YlGnBu')


# counter_lang = {}
# config = ['as_in','bn_in','hi_in','or_in','pa_in','ta_in','te_in']
config = ['bg_bg','cs_cz','et_ee','ka_ge','lv_lv','lt_lt','pl_pl','ro_ro','ru_ru','sk_sk','sl_si','uk_ua']

# ['gu_in','kn_in','ml_in','mr_in','ne_np','sd_in','ur_pk','as_in','bn_in','hi_in','or_in','pa_in','ta_in','te_in']
# config_names = ['gujrati','kannada','malayalam','marathi','nepali','sindhi','urdu','assamese','bengali','hindi','oriya','punjabi','tamil','telugu']
# config_names = ['Catalan','Croatian','Danish','Dutch','AmericanEnglish','Finnish','French','German','Greek','Hungarian','Irish','Italian','LatinAmericanSpanish','Maltese','Portuguese','Swedish','Welsh']
config_names = ['Bulgarian','Czech','Estonian','Georgian','Latvian','Lithuanian','Polish','Romanian','Russian','Slovak','Slovenian','Ukrainian']



sim_mt = np.zeros((len(config),len(config)))
# create the data distribution
for num1,i in enumerate(config):
    for num2,j in enumerate(config):
        proj = torch.load(i+'.pt')
        proj2 = torch.load(j+'.pt')
        # sim_mt[num1][num2] = JSD(proj,proj2)
        sim_mt[num1][num2] = 1.0-distance.jensenshannon(proj,proj2)
        print(f'sim between {i} and {j}:',sim_mt[num1][num2])

 
fig, ax = plt.subplots(figsize=(7,7))
cax = ax.matshow(sim_mt, interpolation='nearest',cmap=cmap)
# ax.grid(True)
plt.title('Similarity matrix')
plt.xticks(range(len(config)), config_names, rotation=90)
plt.yticks(range(len(config)), config_names)
fig.colorbar(cax, ticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8,0.9,1])
plt.show()

plt.subplots(figsize=(5,5))
selected_data = sim_mt
clusters = shc.linkage(selected_data, 
            method='ward', 
            metric="euclidean")
shc.dendrogram(Z=clusters,labels = config_names,orientation='left')
plt.show()