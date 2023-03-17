import torch

pretrained_weights = torch.load("/userhome/cs2/ethanlii/mind-vis/pretrains/GOD/finetuned.pth")

import pdb
pdb.set_trace()

if 'state_dict' in pretrained_weights:
    pretrained_weights = pretrained_weights['state_dict']

print(pretrained_weights)
