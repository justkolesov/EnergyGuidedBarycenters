import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import itertools
import torch.distributions as TD
import torch.utils.data as Tdata
import gc
import os
import pickle
import wandb
import random
from typing import Dict, Any

from tqdm import tqdm
from IPython.display import clear_output
from copy import deepcopy



#========= utils ==========#
def normalize_out_to_0_1(x):
    #assert torch.min(x) < -0.5
    return torch.clip(0.5*(x+1),0,1)


def middle_rgb(x, threshold=0.2):
    
    """
    x - torch.Size([B,3,32,32]) in diapasone [0,1] normalized from Style-GAN and requires_grad
    returns - torch.Size([B,3])
    """
    
    with torch.no_grad():
        # ==== creating_mask ==== 
        x_hsv = matplotlib.colors.rgb_to_hsv(x.permute(0,2,3,1).detach().cpu().numpy()) #[B,32,32,3]
        brightness = torch.from_numpy(x_hsv[:,:,:,-1]) #[B,32,32]
        brightness = brightness.flatten(start_dim=1) #[B, 1024]

        # find indexes with 
        mask = brightness > threshold # [B,1024]
        #==== creating_mask ====
    
    rgb_image = x.flatten(start_dim=2) # [B,3,1024] : requires_grad
    
    # TODO: rewrite in torch form
    output = torch.zeros(rgb_image.shape[0],3).to(x.device)
    for idx,rgb in enumerate(rgb_image): 
        m = mask[idx]
        output[idx][0] = rgb[0][m].mean()
        output[idx][1] = rgb[1][m].mean() 
        output[idx][2] = rgb[2][m].mean() 
    
    return output

def plot_barycenter_map_in_data_space(x1, x2, map_1, map_2, step, n_estimate_points=8):
    
    for distribution, mapping, x in zip(["P","Q"],[map_1,map_2],[x1, x2]):
        fig,ax = plt.subplots(5, n_estimate_points, figsize=(8,5),dpi=150)
        for idx in range(8):
            ax[0,idx].imshow(x[idx].permute(1,2,0).cpu() ,cmap='gray')
            #if  distribution == "P" else ax[0,idx].add_artist(plt.Circle(( 0.5 , 0.5 ), 0.4 ,color=x2[idx].cpu().numpy()) )
            ax[1,idx].imshow(mapping[0][idx].permute(1,2,0).detach().cpu() ,cmap='gray')
            ax[2,idx].imshow(mapping[1][idx].permute(1,2,0).detach().cpu(),cmap='gray')
            ax[3,idx].imshow(mapping[2][idx].permute(1,2,0).detach().cpu()  ,cmap='gray')
            ax[4,idx].imshow(mapping[3][idx].permute(1,2,0).detach().cpu() ,cmap='gray')
            for j in range(5):
                ax[j,idx].set_xticks([]);ax[j,idx].set_yticks([]);
        ax[0,0].set_ylabel(r"$x \sim \mathbb{P}$" if distribution == "P" else r"$x \sim \mathbb{Q}$" )  
        ax[1,0].set_ylabel(r"$T_{1}(x)$")
        ax[3,0].set_ylabel(r"$T_{3}(x)$")
        ax[2,0].set_ylabel(r"$T_{2}(x)$")
        ax[4,0].set_ylabel(r"$T_{4}(x)$")
        fig.tight_layout(pad=0.1)
        wandb.log({f"Barycenter Images of {distribution}":fig},step=step)
        plt.show()
        

def plot_barycenter_map_in_data_space_more(x1, x2, x3, map_1, map_2, map_3, step, n_estimate_points=8):
    
    for distribution, mapping, x in zip(["P","Q","R"],[map_1,map_2,map_3],[x1, x2, x3]):
        fig,ax = plt.subplots(5, n_estimate_points, figsize=(8,5),dpi=150)
        for idx in range(8):
            ax[0,idx].imshow(np.clip(x[idx].permute(1,2,0).cpu(),0,1) )
            ax[1,idx].imshow(np.clip(mapping[0][idx].permute(1,2,0).detach().cpu(),0,1)  )
            ax[2,idx].imshow(np.clip(mapping[1][idx].permute(1,2,0).detach().cpu(),0,1) )
            ax[3,idx].imshow(np.clip(mapping[2][idx].permute(1,2,0).detach().cpu(),0,1)   )
            ax[4,idx].imshow(np.clip(mapping[3][idx].permute(1,2,0).detach().cpu(),0,1)  )
            for j in range(5):
                ax[j,idx].set_xticks([]);ax[j,idx].set_yticks([]);
                
        if distribution == "P":  
            ax[0,0].set_ylabel(r"$x \sim \mathbb{P}$" )  
        elif distribution == "Q":
            ax[0,0].set_ylabel(r"$x \sim \mathbb{Q}$" )
        else:
            ax[0,0].set_ylabel(r"$x \sim \mathbb{R}$" )
            
        ax[1,0].set_ylabel(r"$T_{1}(x)$")
        ax[3,0].set_ylabel(r"$T_{2}(x)$")
        ax[2,0].set_ylabel(r"$T_{3}(x)$")
        ax[4,0].set_ylabel(r"$T_{4}(x)$")
        fig.tight_layout(pad=0.1)
        wandb.log({f"Barycenter Images of {distribution}":fig},step=step)
        plt.show()
        
        
def make_f_pot(idx, nets, config):
    def f_pot(x):
        res = 0.0
        for i, (net, alpha) in enumerate(zip(nets, config.ALPHAS_BARYCENTER)):
            if i == idx:
                res += net(x)
            else:
                res -= alpha * net(x) / (len(config.CLASSES) - 1) / config.ALPHAS_BARYCENTER[idx]
        return res
    return f_pot

def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()    
    
def unfreeze(model):
    for p in model.parameters():
        p.requires_grad_(True)
    model.train(True)
#========== utils ===========#
 
    
    
    
def _default_input_preprocessing(X):
    if len(X.shape) == 1:
        X = X.view(-1, 1)
    X.requires_grad_(True)
    return X

def id_pretrain_model(
    model, sampler, lr=1e-3, n_max_iterations=2000, n_verbose=1000,
    batch_size=1024, loss_stop=1e-3, verbose=True, true_x_transform = lambda x: x,
    logit_postprocessing = lambda x: x,
    input_preprocessing = _default_input_preprocessing):
    '''
    Pretrain the model to be id function
    '''

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-8)
    for it in tqdm(range(n_max_iterations), disable = not verbose):
        X = sampler.sample(batch_size)

        with torch.no_grad():
            Y_true = true_x_transform(X)

        X = input_preprocessing(X)
        Y = logit_postprocessing(model(X))
        loss = F.mse_loss(Y, Y_true.reshape_as(Y))
        # loss = F.l1_loss(model(X), Y)
        loss.backward()

        opt.step()
        opt.zero_grad() 

        if verbose:
            if it % n_verbose == n_verbose-1:
                clear_output(wait=True)
                print('Loss:', loss.item())

        if loss.item() < loss_stop:
            clear_output(wait=True)
            print('Final loss:', loss.item())
            break
    return model

def clean_resources(*tsrs):
    del tsrs
    gc.collect()
    torch.cuda.empty_cache()

class LabeledDataset:

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, i):
        return (self.data[i], self.labels[i])

    def __len__(self):
        return len(self.data)

class GenSampler:

    def __init__(self, Zdistrib, G):
        self.Zdistrib = Zdistrib
        self.G = G

    def sample(self, size, Z_no_grad=True, G_no_grad=False):
        if not isinstance(size, tuple):
            size = (size,)
        if Z_no_grad:
            with torch.no_grad():
                Z = self.Zdistrib.sample(size)
        else:
            Z = self.Zdistrib.sample(size)
        if G_no_grad:
            with torch.no_grad():
                X = self.G(Z)
        else:
            X = self.G(Z)
        return X


class Distrib2Sampler:

    def __init__(self, distrib):
        self.distrib = distrib

    def sample(self, size):
        if not isinstance(size, tuple):
            size = (size,)
        return self.distrib.sample(size)


class JointSampler:

    def __init__(self, *samplers):
        self.samplers = samplers

    def sample(self, batch_size):
        return [sampler.sample(batch_size) for sampler in self.samplers]


def make_numpy(X):
    if isinstance(X, torch.Tensor):
        return X.detach().cpu().numpy()
    if isinstance(X, np.ndarray):
        return X
    return np.asarray(X)

def batch_jacobian(x, y, create_graph=True, retain_graph=True):
    """Computes the Jacobian of f w.r.t x.
    :param f: function R^N -> R^N
    :param x: torch.tensor of shape [B, N]
    :return: Jacobian matrix (torch.tensor) of shape [B, N, N]
    """
    B, N = y.shape
    jacobian = list()
    for i in range(N):
        v = torch.zeros_like(y)
        v[:, i] = 1.
        dy_i_dx = autograd.grad(y, x, grad_outputs=v, retain_graph=True, create_graph=True, allow_unused=True)[0]  # shape [B, N]
        jacobian.append(dy_i_dx)
    jacobian = torch.stack(jacobian, dim=2).requires_grad_()
    return jacobian

def computePotGrad(input, output, create_graph=True, retain_graph=True):
    '''
    :Parameters:
    input : tensor (bs, *shape)
    output: tensor (bs, 1) , NN(input)
    :Returns:
    gradient of output w.r.t. input (in batch manner), shape (bs, *shape)
    '''
    grad = autograd.grad(
        outputs=output, 
        inputs=input,
        grad_outputs=torch.ones_like(output),
        create_graph=create_graph,
        retain_graph=retain_graph,
    ) # (bs, *shape) 
    return grad[0]


class DataLoaderWrapper:
    '''
    Helpful class for using the 
    DistributionSampler's in torch's 
    DataLoader manner
    '''

    class DummyDataset:

        def __init__(self, bs, n_batch):
            self._len = bs * n_batch

        def __len__(self):
            return self._len

        def __getitem__(self, pos):
            raise NotImplementedError()

    class FiniteRepeatDSIterator:

        def __init__(self, sampler, batch_size, n_batches):
            # dataset = sampler.sample(batch_size * n_batches)
            # assert(len(dataset.shape) >= 2)
            # new_size = (n_batches, batch_size) + dataset.shape[1:]
            # self.dataset = dataset.view(new_size)
            self.sampler = sampler
            self.dataset = []
            self.batch_size = batch_size
            self.n_batches = n_batches

        def __iter__(self):
            for i in range(self.n_batches):
                if i >= len(self.dataset):
                    sample = self.sampler.sample(self.batch_size)
                    self.dataset.append(sample)
                    yield sample
                else:
                    yield self.dataset[i]

    class FiniteUpdDSIterator:

        def __init__(self, sampler, batch_size, n_batches):
            self.sampler = sampler
            self.batch_size = batch_size
            self.n_batches = n_batches

        def __iter__(self):
            for i in range(self.n_batches):
                yield self.sampler.sample(self.batch_size)

    class InfiniteDsIterator:

        def __init__(self, sampler, batch_size):
            self.sampler = sampler
            self.batch_size = batch_size

        def __iter__(self):
            return self

        def __next__(self):
            return self.sampler.sample(self.batch_size)


    def __init__(self, sampler, batch_size, n_batches=None, store_dataset=False):
        '''
        n_batches : count of batches before stop_iterations, if None, the dataset is infinite
        store_datset : if n_batches is not None and store_dataset is True, 
        during the first passage through the dataset the data will be stored,
        and all other epochs will use the same dataset, stored during the first pass
        '''
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.store_dataset = store_dataset
        self.sampler = sampler

        if self.n_batches is None:
            self.ds_iter = DataLoaderWrapper.InfiniteDsIterator(
                sampler, self.batch_size)
            return

        if not self.store_dataset:
            self.ds_iter = DataLoaderWrapper.FiniteUpdDSIterator(
                sampler, self.batch_size, self.n_batches)
            return

        self.ds_iter = DataLoaderWrapper.FiniteRepeatDSIterator(
            sampler, self.batch_size, self.n_batches)


    def __iter__(self):
        return iter(self.ds_iter)
    
    @property
    def dataset(self):
        return DataLoaderWrapper.DummyDataset(self.batch_size, self.n_batches)

#####################################################
## taken from W2GN https://github.com/iamalexkorotin/Wasserstein2GenerativeNetworks

def energy_based_distance(X, Y, n_projections=10000, biased=False, log_scale=False, log_eps=1e-8):
    '''
    An implementation of unbiased energy-based distance between
    two disributions given by i.i.d. batches.
    This implementation computes an unbiased sliced continuous
    ranking probability score (via random projections).
    It equals energy based distance up to a multiplicative
    constant depending on the dimension,
    see Theorem 4.1 of https://arxiv.org/pdf/1912.07048.pdf for details 
    '''
    assert X.size(1) == Y.size(1)

    thetas = torch.randn(n_projections, X.size(1)).to(X.device)
    thetas = thetas / thetas.norm(2, dim=1, keepdim=True)
    
    # Sorted projection of joint matrix and reverse sorted index
    pXY, idx = torch.sort(thetas @ torch.cat((X, Y), dim=0).transpose(0,1), dim=1)
    
    # Normalized indicator functions (1./X.size(0) for elements from X, -1./Y.size(0) for Y)
    I = torch.ones(idx.size(), dtype=torch.float32, device=X.device) / X.size(0)
    I[idx >= X.size(0)] = -1. / Y.size(0)
    
    SFXY = torch.cumsum(I, dim=1)
    scrps_biased = torch.mean(torch.sum((pXY[:, 1:] - pXY[:, :-1]) * SFXY[:, :-1] ** 2, dim=1))
    
    if biased:
        return scrps_biased
    
    pX_mask = idx < X.size(0)
    SFX = torch.cumsum(I[pX_mask].reshape(-1, X.size(0)), dim=1)
    pX = pXY[pX_mask].reshape(-1, X.size(0))
    var_SFX = torch.mean(torch.sum((pX[:, 1:] - pX[:, :-1]) * SFX[:, :-1] * (1. - SFX[:, :-1]), dim=1)) / (X.size(0) - 1)
    
    pY_mask = idx >= X.size(0)
    SFY = torch.cumsum(I[pY_mask].reshape(-1, Y.size(0)), dim=1)
    pY = pXY[pY_mask].reshape(-1, Y.size(0))
    var_SFY = torch.mean(torch.sum((pY[:, 1:] - pY[:, :-1]) * SFY[:, :-1] * (1. - SFY[:, :-1]), dim=1)) / (Y.size(0) - 1)
    
    res = scrps_biased - var_SFX - var_SFY
    if not log_scale:
        return res
    else:
        return torch.log(res + log_eps)

##################################
## Config with model parameters

class Config():

    @staticmethod
    def fromdict(config_dict):
        config = Config()
        for name, val in config_dict.items():
            setattr(config, name, val)
        return config
    
    @staticmethod
    def load(path):
        os.makedirs(os.path.join(*("#" + path).split('/')[:-1])[1:], exist_ok=True)
        with open(path, 'rb') as handle:
            config_dict = pickle.load(handle)
        return Config.fromdict(config_dict)

    def store(self, path):
        os.makedirs(os.path.join(*("#" + path).split('/')[:-1])[1:], exist_ok=True)
        with open(path, 'wb') as handle:
            pickle.dump(self.__dict__, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def set_attributes(
            self, 
            attributes_dict: Dict[str, Any], 
            require_presence : bool = True,
            keys_upper: bool = True
        ) -> int:
        _n_set = 0
        for attr, val in attributes_dict.items():
            if keys_upper:
                attr = attr.upper()
            set_this_attribute = True
            if require_presence:
                if not attr in self.__dict__.keys():
                    set_this_attribute = False
            if set_this_attribute:
                if isinstance(val, list):
                    val = tuple(val)
                setattr(self, attr, val)
                _n_set += 1
        return _n_set


def get_changing_values_range(size, val_init, val_fin, coldstart=0, coldfin=0, progression='lin'):
    assert progression in ['lin', 'geom']
    assert size >= coldstart + coldfin
    coldstart_range = np.array([val_init,]*coldstart)
    coldfin_range = np.array([val_fin,]*(coldfin + 1))
    if progression == 'lin':
        change_range = np.linspace(val_init, val_fin, num=(size - coldstart - coldfin))
    else:
        assert val_init > 0
        assert val_fin > 0
        change_range = np.geomspace(val_init, val_fin, num=(size - coldstart - coldfin))
    res_range = np.concatenate((coldstart_range, change_range, coldfin_range), axis=0)
    assert len(res_range) >= size + 1
    return res_range[:size + 1]

class ParametersSpecificator:

    def __init__(self, default, specification=None):
        '''
        specification looks like 
        {
            (key1, key2, keyk): value, 
            ...
        }
        '''
        self.default = default
        if specification is not None:
            old_specification = deepcopy(specification)
            for key, value in old_specification.items():
                if not isinstance(key, tuple):
                    del specification[key]
                    specification[(key,)] = value
            self.specification = specification
        else:
            self.specification = dict()
    
    def upd_specification(self, specification):
        for key, value in specification.items():
            if not isinstance(key, tuple):
                key = (key,)
            self.specification[key] = value
    
    def upd_default(self, default):
        self.default = default

    def __call__(self, *keys):
        if not keys in self.specification.keys():
            return self.default
        return self.specification[keys]


class DataParallelAttrAccess(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
        
        
 
        
        
        
        