import torch
import pdb
pdb.set_trace()
origin_sd = torch.load('/home/hanqingli/Mind-V/pretrains/ldm/label2img/model.ckpt', map_location="cpu")['state_dict']
sd_control = torch.load('/home/hanqingli/Mind-V/pretrains/ldm/label2img/v1-5-pruned.ckpt', map_location="cpu")['state_dict']