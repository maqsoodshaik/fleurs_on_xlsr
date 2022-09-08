import matplotlib.pyplot as plt
import torch
import numpy as np
config = ['gu_in','kn_in','ml_in','mr_in','ne_np','sd_in','ur_pk','as_in','bn_in','hi_in','or_in','pa_in','ta_in','te_in','ca_es','hr_hr','da_dk','nl_nl','en_us','fi_fi','fr_fr','de_de','el_gr','hu_hu','ga_ie','it_it','es_419','mt_mt','pt_br','sv_se','cy_gb','bg_bg','cs_cz','et_ee','ka_ge','lv_lv','lt_lt','pl_pl','ro_ro','ru_ru','sk_sk','sl_si','uk_ua','af_za','am_et','ff_sn','ha_ng','ig_ng','kam_ke','ln_cd','luo_ke','nso_za','ny_mw','om_et','sn_zw','so_so','umb_ao','wo_sn','xh_za','yo_ng','ar_eg','kk_kz','ky_kg','mn_mn','ps_af','fa_ir','tg_tj','tr_tr','ast_es','bs_ba','gl_es','is_is','kea_cv','lb_lu','nb_no','oc_fr','hy_am','be_by','mk_mk','sr_rs','az_az','he_il','ckb_iq','uz_uz','fil_ph','jv_id','km_kh','ms_my','mi_nz','my_mm','ko_kr','ceb_ph','id_id','lo_la','th_th','vi_vn','lg_ug','sw_ke','zu_za','yue_hant_hk','ja_jp','cmn_hans_cn']
def des(name):
    proj = torch.load(name+'.pt')
    k = np.array(proj)
    # print(len(np.nonzero(k)[0]))
    proj_fil = np.nonzero(k)[0]
    proj,ids = torch.sort(proj,descending=True)
    # proj_fil = proj[k]
    return proj,ids,proj_fil
proj_fil_list = np.array([])
for i in config:
    p, id,proj_fil= des(i)
    proj_fil_list = np.concatenate((proj_fil_list,proj_fil))
p = np.array(proj_fil_list).flatten().squeeze()
print(len(set(p)))
    # plt.plot(p)
    # print(f'{i}:{id[0:11]}')
    

# plt.show()