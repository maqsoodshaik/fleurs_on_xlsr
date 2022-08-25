import matplotlib.pyplot as plt
import torch
config = ['en_us','de_de','fr_fr','ar_eg','it_it','ca_es','es_419','ar_eg','kk_kz','ky_kg','mn_mn','ps_af','fa_ir','tg_tj','tr_tr']
def des(name):
    proj = torch.load(name+'.pt')
    proj,ids = torch.sort(proj,descending=True)
    return proj,ids
for i in config:
    p, id= des(i)
    plt.plot(p)
    print(f'{i}:{id[0:11]}')

plt.show()