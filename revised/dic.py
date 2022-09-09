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

cmap = cm.get_cmap("YlGnBu")


def des(name):
    # proj = torch.load(name+'.pt')
    k = np.array(name)
    # print(len(np.nonzero(k)[0]))
    k = np.nonzero(k)[0]
    # proj,ids = torch.sort(proj,descending=True)
    proj_fil = proj[k]
    return proj_fil


# counter_lang = {}
# config = ['gu_in','kn_in','ml_in','mr_in','ne_np','sd_in','ur_pk','as_in','bn_in','hi_in','or_in','pa_in','ta_in','te_in','ca_es','hr_hr','da_dk','nl_nl','en_us','fi_fi','fr_fr','de_de','el_gr','hu_hu','ga_ie','it_it','es_419','mt_mt','pt_br','sv_se','cy_gb','bg_bg','cs_cz','et_ee','ka_ge','lv_lv','lt_lt','pl_pl','ro_ro','ru_ru','sk_sk','sl_si','uk_ua','af_za','am_et','ff_sn','ha_ng','ig_ng','kam_ke','ln_cd','luo_ke','nso_za','ny_mw','om_et','sn_zw','so_so','umb_ao','wo_sn','xh_za','yo_ng','ar_eg','kk_kz','ky_kg','mn_mn','ps_af','fa_ir','tg_tj','tr_tr','ast_es','bs_ba','gl_es','is_is','kea_cv','lb_lu','nb_no','oc_fr','hy_am','be_by','mk_mk','sr_rs','az_az','he_il','ckb_iq','uz_uz','fil_ph','jv_id','km_kh','ms_my','mi_nz','my_mm','ko_kr','ceb_ph','id_id','lo_la','th_th','vi_vn','lg_ug','sw_ke','zu_za','yue_hant_hk','ja_jp','cmn_hans_cn']
config = [
    "bg_bg",
    "cs_cz",
    "et_ee",
    "ka_ge",
    "lv_lv",
    "lt_lt",
    "pl_pl",
    "ro_ro",
    "ru_ru",
    "sk_sk",
    "sl_si",
    "uk_ua",
    "gu_in",
    "kn_in",
    "ml_in",
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
# config_names = ['gujrati','kannada','malayalam','marathi','nepali','sindhi','urdu','assamese','bengali','hindi','oriya','punjabi','tamil','telugu','Catalan','Croatian','Danish','Dutch','American English','Finnish','French','German','Greek','Hungarian','Irish','Italian','Latin American Spanish','Maltese','Portuguese','Swedish','Welsh','Bulgarian','Czech','Estonian','Georgian','Latvian','Lithuanian','Polish','Romanian','Russian','Slovak','Slovenian','Ukrainian','Afrikaans','Amharic','Fula ','Hausa','Igbo','Kamba','Lingala','Luo','Northern-Sotho','Nyanja','Oromo ','Shona',' Somali','Umbundu','Wolof','Xhosa','Yoruba','Arabic','Kazakh',' Kyrgyz','Mongolian','Pashto','Persian','Tajik','Turkish','Asturian','Bosnian','Galician','Icelandic','Kabuverdianu','Luxembourgish','Norwegian','Occitan','Armenian',' Belarusian','Macedonian','Serbian','Azerbaijani','Hebrew ','Sorani-Kurdish','Uzbek','Filipino','Javanese','Khmer','Malay',' Maori','Burmese','Korean','Cebuano','Indonesian','Lao','Thai','Vietnamese','Ganda','Swahili','Zulu','Cantonese','Japanese','Mandarin']
config_names = [
    "Bulgarian",
    "Czech",
    "Estonian",
    "Georgian",
    "Latvian",
    "Lithuanian",
    "Polish",
    "Romanian",
    "Russian",
    "Slovak",
    "Slovenian",
    "Ukrainian",
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


sim_mt = np.zeros((len(config), len(config)))
# create the data distribution
for num1, i in enumerate(config):
    for num2, j in enumerate(config):
        proj = torch.load(i + ".pt")
        proj2 = torch.load(j + ".pt")
        # sim_mt[num1][num2] = JSD(proj,proj2)
        # sim_mt[num1][num2] = 1.0-distance.jensenshannon(proj+10**-6,proj2+10**-6)
        # proj = des(proj)
        # proj2 = des(proj2)
        sim_mt[num1][num2] = 1.0 - distance.jensenshannon(proj, proj2)
        print(f"sim between {i} and {j}:", sim_mt[num1][num2])


fig, ax = plt.subplots(figsize=(7, 7))
cax = ax.matshow(sim_mt, interpolation="nearest", cmap=cmap)
# ax.grid(True)
plt.title("Similarity matrix")
plt.xticks(range(len(config)), config_names, rotation=90)
plt.yticks(range(len(config)), config_names)
fig.colorbar(cax, ticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.savefig("sim.pdf", bbox_inches="tight")
plt.show()

plt.subplots(figsize=(5, 5))
selected_data = sim_mt
clusters = shc.linkage(selected_data, method="ward", metric="euclidean")
shc.dendrogram(Z=clusters, labels=config_names, orientation="left")
plt.savefig("lang.pdf", bbox_inches="tight")
plt.show()
