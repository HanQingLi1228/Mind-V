import torch

before_finetune = "/home/hanqingli/Mind-V/results/generation/14-04-2023-12-47-28"

after_finetune = "/home/hanqingli/Mind-V/results/generation/13-04-2023-07-14-52"

before_model = torch.load(before_finetune, map_location="cpu")

after_model = torch.load(after_finetune, map_location="cpu")

print(before_model['model_state_dict'])
print(after_model['model_state_dict'])
