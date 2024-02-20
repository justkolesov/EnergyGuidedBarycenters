import torch
import torchvision
from src.utils import normalize_out_to_0_1, computePotGrad, middle_rgb, Config
from typing import Callable, Tuple, Union


def cost_l2_grad_y(y, x):
    '''
    y - torch.Size([B,C,H,W]),  requires_grad = No
    x - torch.Size([B,C,H,W]),  requires_grad = No
    returns \nabla_y c(x, y)=0.5||x-y||^{2}_{2}
    '''
    return y - x



def cond_score(
        f : Callable[[torch.Tensor], torch.Tensor], 
        cost_grad_y_fn : Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
        y : torch.Tensor, 
        x : torch.Tensor,
        config: Config,
        flag_grayscale=False,
        flag_f_G_latent=False,
        latent2data_gen=None,
        ret_stats=False
        
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    
    trnsfm = torchvision.transforms.Grayscale() if flag_grayscale else torchvision.transforms.Lambda(lambda x:x)
    
    with torch.enable_grad():
        y.requires_grad_(True) 
        #proto_s = f(trnsfm(normalize_out_to_0_1(latent2data_gen(y, c=None)))) if flag_f_G_latent else f(y)
        proto_s = f( normalize_out_to_0_1(latent2data_gen(y, c=None)) ) if flag_f_G_latent else f(y)
        s = computePotGrad(y, proto_s)
        assert s.shape == y.shape
        
    cost_coeff = config.LANGEVIN_COST_COEFFICIENT * (config.LANGEVIN_SAMPLING_NOISE ** 2 / config.HREG)
    cost_part = cost_grad_y_fn(y, x) * cost_coeff if latent2data_gen==None else cost_grad_y_fn(y, x, latent2data_gen, config,
                                                                                               flag_grayscale) * cost_coeff
    
                                                                                            
    score_part = s * config.LANGEVIN_SCORE_COEFFICIENT
    if not ret_stats:
        return score_part - cost_part
    return score_part - cost_part, cost_part, score_part


#============ Costs for Image experimentts =========#

def cost_grad_image_latent(z, x, latent2data_gen, config, flag_grayscale=False):
    '''
    z - torch.Size([B,latent_dim]), requires_grad = No
    x - torch.Size([B,C,H,W]),  requires_grad = No
    returns \nabla_z c(x, z)=0.5||x-G(z)||^{2}_{2}
    returns - torch.Size([B,latent_dim])
    '''
    with torch.enable_grad():
        z.requires_grad_(True)
        cost = 0.5 * torch.flatten( normalize_out_to_0_1(latent2data_gen(z, c=None)) - x,
                                   start_dim=1).pow(2).mean(dim=1, keepdim=True)
        assert cost.shape == torch.Size([z.size(0), 1])
        res = computePotGrad(z, cost)
    return res


def cost_grad_image_shape_latent(z, x, latent2data_gen, config, flag_grayscale):
    """
    z - torch.Size([B,latent_dim]), requires_grad = No
    x - torch.Size([B,C,H,W]),  requires_grad = No
    returns \nabla_z c(x, z)=0.5||x-G(z)||^{2}_{2}
    returns - torch.Size([B,latent_dim])
    """
    # TODO: torchvision.transfrom.identity ?
    trnsfm = torchvision.transforms.Grayscale() if flag_grayscale else torchvision.transforms.Lambda(lambda x:x)
    
    with torch.enable_grad():
        z.requires_grad_(True)
        """
        cost = 0.5 * torch.flatten( trnsfm(normalize_out_to_0_1(latent2data_gen(z, c=None))) - x,
                                   start_dim=1).pow(2).sum(dim=1, keepdim=True)
        """
        cost =  0.5 * torch.flatten( torch.max( normalize_out_to_0_1(latent2data_gen(z, c=None)),dim=1,keepdim=True)[0] - x,
                                   start_dim=1).pow(2).mean(dim=1, keepdim=True)
        assert cost.shape == torch.Size([z.size(0), 1])
        res = computePotGrad(z, cost)
    return res


def cost_grad_image_color_latent(z, x, latent2data_gen, config, flag_grayscale=False):
    """
    z - torch.Size([B,latent_dim]), requires_grad = No
    x - torch.Size([B,3]),  requires_grad = No
    returns \nabla_z c(x, z)=0.5||x-G(z)||^{2}_{2}
    returns - torch.Size([B,3])
    """
    
    with torch.enable_grad():
        z.requires_grad_(True)
        cost = 0.5 * torch.flatten( middle_rgb(normalize_out_to_0_1( latent2data_gen(z, c=None) )) - x,
                                   start_dim=1).pow(2).mean(dim=1, keepdim=True)
        assert cost.shape == torch.Size([z.size(0), 1])
        res = computePotGrad(z, config.COST_COLOR_SCALER*cost)
        
    return res


