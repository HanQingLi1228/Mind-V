import torch

before_finetune = "/data/hanqingli/diffusers/origin"

#after_finetune = "/data/hanqingli/pretrains/GOD"
after_finetune = "/data/hanqingli/diffusers/after"
before_model = torch.load(before_finetune + '/lora_unet.pth', map_location="cpu")

after_model = torch.load(after_finetune + '/lora_unet.pth', map_location="cpu")

#print(before_model['model_state_dict'])
#print(after_model['model_state_dict'])
import pdb
pdb.set_trace()
for key1 in before_model.keys():
    if not (before_model[key1].equal(after_model[key1])):
        print(key1)
        
    


