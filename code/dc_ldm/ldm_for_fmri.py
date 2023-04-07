import numpy as np
import wandb
import torch
from dc_ldm.util import instantiate_from_config
from omegaconf import OmegaConf
import torch.nn as nn
import os
from dc_ldm.models.diffusion.plms import PLMSSampler
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sc_mbm.mae_for_fmri import fmri_encoder
from dc_ldm.modules.encoders.modules import FrozenCLIPEmbedder

def create_model_from_config(config, num_voxels, global_pool):
    model = fmri_encoder(num_voxels=num_voxels, patch_size=config.patch_size, embed_dim=config.embed_dim,
                depth=config.depth, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio, global_pool=global_pool) 
    return model

class control_stage_model(nn.Module):
    def __init__(self, metafile, num_voxels, cond_dim=1280, global_pool=True):
        super().__init__()
        # prepare pretrained fmri mae 
        model = create_model_from_config(metafile['config'], num_voxels, global_pool)
        model.load_checkpoint(metafile['model'])
        self.mae = model
        self.fmri_seq_len = model.num_patches
        self.fmri_latent_dim = model.embed_dim
        if global_pool == False:
            self.channel_mapper = nn.Sequential(
                nn.Conv1d(self.fmri_seq_len, self.fmri_seq_len // 2, 1, bias=True),
                nn.Conv1d(self.fmri_seq_len // 2, 77, 1, bias=True)
            )
        self.dim_mapper = nn.Linear(self.fmri_latent_dim, cond_dim, bias=True)
        self.global_pool = global_pool

    def forward(self, x):
        import pdb
        pdb.set_trace()
        # n, c, w = x.shape
        latent_crossattn = self.mae(x)
        if self.global_pool == False:
            latent_crossattn = self.channel_mapper(latent_crossattn)
        latent_crossattn = self.dim_mapper(latent_crossattn)
        out = latent_crossattn
        return out

class fLDM:

    def __init__(self, metafile, num_voxels, device=torch.device('cpu'),
                 pretrain_root='../pretrains/ldm/label2img',
                 logger=None, ddim_steps=250, global_pool=True, use_time_cond=True):
        self.ckp_path = os.path.join(pretrain_root, 'mind-vis-add-control_sd2.ckpt')
        self.config_path = '/home/hanqingli/Mind-V/code/custom/config_custom_control.yaml' 
        config = OmegaConf.load(self.config_path)
        config.model.params.unet_config.params.use_time_cond = use_time_cond
        config.model.params.unet_config.params.global_pool = global_pool

        self.cond_dim = config.model.params.unet_config.params.context_dim

        model = instantiate_from_config(config.model)
        #import pdb
        #pdb.set_trace()
        pl_sd = torch.load(self.ckp_path, map_location="cpu")
        # import pdb
        # pdb.set_trace()

        m, u = model.load_state_dict(pl_sd, strict=False)
        #model.cond_stage_trainable = True
        model.cond_stage_trainable = False
        #model.cond_stage_model = cond_stage_model(metafile, num_voxels, self.cond_dim, global_pool=global_pool)
        model.control_stage_model = control_stage_model(metafile, num_voxels, self.cond_dim, global_pool=global_pool)

        model.ddim_steps = ddim_steps
        model.re_init_ema()
        if logger is not None:
            logger.watch(model, log="all", log_graph=False)

        model.p_channels = config.model.params.channels
        model.p_image_size = config.model.params.image_size
        model.ch_mult = config.model.params.first_stage_config.params.ddconfig.ch_mult
        
        self.device = device    
        self.model = model
        self.ldm_config = config
        self.pretrain_root = pretrain_root
        self.fmri_latent_dim = model.control_stage_model.fmri_latent_dim
        self.metafile = metafile

    def finetune(self, trainers, dataset, test_dataset, bs1, lr1,
                output_path, config=None):
        config.trainer = None
        config.logger = None
        self.model.main_config = config
        self.model.output_path = output_path
        # self.model.train_dataset = dataset
        self.model.run_full_validation_threshold = 0.15
        # stage one: train the cond encoder with the pretrained one
        #ffff = open("/userhome/cs2/ethanlii/mind-vis/model.txt", "w")
        #ffff.write(self.model)
        #print(self.model)
        #import pdb
        #pdb.set_trace()
        # # stage one: only optimize conditional encoders
        print('\n##### Stage One: only optimize conditional encoders #####')
        dataloader = DataLoader(dataset, batch_size=bs1, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

        '''before_need_grad = []
        before_no_need_grad = []
        for names, param in self.model.named_parameters():
            if param.requires_grad == True:
                before_need_grad.append(names)
            elif param.requires_grad == False:
                before_no_need_grad.append(names)
            else:
                print(names)  '''      

        self.model.unfreeze_whole_model()
        self.model.freeze_diffusion_model()
        self.model.freeze_cond_stage()
        self.model.freeze_first_stage()
        # import pdb
        # pdb.set_trace()
        self.model.learning_rate = lr1
        self.model.train_cond_stage_only = True
        self.model.eval_avg = config.eval_avg
        #import pdb
        #pdb.set_trace()
        # need_grad = []
        # no_need_grad = []
        # for names, param in self.model.named_parameters():
        #     if param.requires_grad == True:
        #         need_grad.append(names)
        #     elif param.requires_grad == False:
        #         no_need_grad.append(names)
        #     else:
        #         print(names)
        # import pdb
        # pdb.set_trace()

        '''aaa = 0
        bbb = 0
        totall = 0
        for param in self.model.parameters():
            if param.requires_grad == True:
                aaa += param.numel()
            elif param.requires_grad == False:
                bbb += param.numel()
            totall += param.numel()'''
        import pdb
        pdb.set_trace()
        # 可训练参数（just control_model(COntrolNet) and Control_stage_model(MAE)） -> 641 keys
        # compare origin mindvis : (cond_stage_model(MAE) + diffusion_model) -> 995 keys
        trainers.fit(self.model, dataloader, val_dataloaders=test_loader)
        

        self.model.unfreeze_whole_model()
        
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'config': config,
                'state': torch.random.get_rng_state()

            },
            os.path.join(output_path, 'checkpoint.pth')
        )
        

    @torch.no_grad()
    def generate(self, fmri_embedding, num_samples, ddim_steps, HW=None, limit=None, state=None):
        # fmri_embedding: n, seq_len, embed_dim
        all_samples = []
        if HW is None:
            shape = (self.ldm_config.model.params.channels, 
                self.ldm_config.model.params.image_size, self.ldm_config.model.params.image_size)
        else:
            num_resolutions = len(self.ldm_config.model.params.first_stage_config.params.ddconfig.ch_mult)
            shape = (self.ldm_config.model.params.channels,
                HW[0] // 2**(num_resolutions-1), HW[1] // 2**(num_resolutions-1))

        model = self.model.to(self.device)
        sampler = PLMSSampler(model)
        # sampler = DDIMSampler(model)
        if state is not None:
            torch.cuda.set_rng_state(state)
            
        with model.ema_scope():
            model.eval()
            #import pdb
            #pdb.set_trace()
            for count, item in enumerate(fmri_embedding):
                if limit is not None:
                    if count >= limit:
                        break
                fmri = item['fmri']
                gt_image = rearrange(item['image'], 'h w c -> 1 c h w') # h w c
                
                txt = item['txt']
                txt_latent = []
                for i in range(num_samples):
                    txt_latent.append(txt)
                c = model.get_learned_conditioning(txt_latent)
                c.to(self.device)

                print(f"rendering {num_samples} examples in {ddim_steps} steps.")
                # assert latent.shape[-1] == self.fmri_latent_dim, 'dim error'
                
                fmri = model.get_control_feature(repeat(fmri, 'h w -> c h w', c=num_samples).to(self.device))
                condition = dict(c_crossattn=[c], c_concat=[fmri])
                samples_ddim, _ = sampler.sample(S=ddim_steps, 
                                                conditioning=condition,
                                                batch_size=num_samples,
                                                shape=shape,
                                                verbose=False)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                gt_image = torch.clamp((gt_image+1.0)/2.0, min=0.0, max=1.0)
                gt_image = F.interpolate(gt_image, scale_factor=2, mode='nearest')
                all_samples.append(torch.cat([gt_image, x_samples_ddim.detach().cpu()], dim=0)) # put groundtruth at first
                
        
        # display as grid
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=num_samples+1)

        # to image
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        model = model.to('cpu')
        
        return grid, (255. * torch.stack(all_samples, 0).cpu().numpy()).astype(np.uint8)


