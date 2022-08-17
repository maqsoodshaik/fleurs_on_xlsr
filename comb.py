import torch
proj = torch.load('ml_in.pt')
proj_2 = torch.load('ml_in_2.pt')
proj_t = torch.cat((proj,proj_2))
torch.save(proj_t, f'ml_in_t.pt')
