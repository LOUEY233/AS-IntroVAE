import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from copy import deepcopy
from collections import OrderedDict
from torch import Tensor
from sys import stderr
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST, SVHN, LSUN
from torchvision import transforms
from PIL import Image
import math

# standard
import os
import random
import time
import numpy as np
from tqdm import tqdm
import pickle
from dataset import ImageDatasetFromFile, DigitalMonstersDataset
from models import SoftIntroVAE,reparameterize

def get_dataset(dataset,batch_size,num_workers):
# --------------build models -------------------------
    if dataset == 'cifar10':
        image_size = 32
        channels = [64, 128, 256]
        train_set = CIFAR10(root='./dataset/cifar10_ds', train=True, transform=transforms.ToTensor())
        ch = 3
    elif dataset == 'celeb128':
        channels = [64, 128, 256, 512, 512]
        image_size = 128
        ch = 3
        output_height = 128
        data_root = './dataset/celeba128/img_align_celeba'
        train_list = [x for x in os.listdir(data_root) if is_image_file(x)]
        assert len(train_list) > 0
        train_set = ImageDatasetFromFile(train_list, data_root, input_height=None, crop_height=None,
                                         output_height=output_height, is_mirror=True)
    elif dataset == 'oxford':
        channels = [64, 128, 256, 512, 512]
        image_size = 128
        ch = 3
        output_height = 128
        data_root = './dataset/oxford'
        train_list = [x for x in os.listdir(data_root) if is_image_file(x)]
        assert len(train_list) > 0
        train_set = ImageDatasetFromFile(train_list, data_root, input_height=None, crop_height=None,
                                         output_height=output_height, is_mirror=True)
    elif dataset == 'celeb256':
        channels = [64, 128, 256, 512, 512, 512]
        image_size = 256
        ch = 3
        output_height = 256
        data_root = './dataset/celeba256/img_align_celeba'
        train_list = [x for x in os.listdir(data_root) if is_image_file(x)]
        assert len(train_list) > 0
        train_set = ImageDatasetFromFile(train_list, data_root, input_height=None, crop_height=None,
                                         output_height=output_height, is_mirror=True)
    # elif dataset == 'celeb1024':
    #     channels = [16, 32, 64, 128, 256, 512, 512, 512]
    #     image_size = 1024
    #     ch = 3
    #     output_height = 1024
    #     train_size = 29000
    #     data_root = './' + dataset
    #     image_list = [x for x in os.listdir(data_root) if is_image_file(x)]
    #     train_list = image_list[:train_size]
    #     assert len(train_list) > 0

    #     train_set = ImageDatasetFromFile(train_list, data_root, input_height=None, crop_height=None,
    #                                      output_height=output_height, is_mirror=True)
    # elif dataset == 'monsters128':
    #     channels = [64, 128, 256, 512, 512]
    #     image_size = 128
    #     ch = 3
    #     data_root = '.dataset/monsters_ds/'
    #     train_set = DigitalMonstersDataset(root_path=data_root, output_height=image_size)
    # elif dataset == 'svhn':
    #     image_size = 32
    #     channels = [64, 128, 256]
    #     train_set = SVHN(root='./dataset/svhn', split='train', transform=transforms.ToTensor(), download=True)
    #     ch = 3
    elif dataset == 'lsun':
        channels = [64, 128, 256, 512, 512]
        image_size = 128
        ch = 3
        output_height = 128
        train_set = LSUN(root='./dataset/lsun', classes=['classroom_train'], transform=transforms.Compose([transforms.ToTensor(),
                                                                            transforms.Resize(size=(output_height, output_height))]))
    # elif dataset == 'fmnist':
    #     image_size = 28
    #     channels = [64, 128]
    #     train_set = FashionMNIST(root='./dataset/fmnist_ds', train=True, download=True, transform=transforms.ToTensor())
    #     ch = 1
    elif dataset == 'mnist':
        image_size = 28
        channels = [64, 128]
        train_set = MNIST(root='./dataset/mnist_ds', download=True, transform=transforms.ToTensor())
        ch = 1
    else:
        raise NotImplementedError("dataset is not supported")
    
    train_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                   num_workers=num_workers)
    
    return ch, channels,image_size,train_data_loader 

def get_model(z_dim,pretrained,ch,channels, image_size,device):
    
    model = SoftIntroVAE(cdim=ch, zdim=z_dim, channels=channels, image_size=image_size).to(device)

    if pretrained is not None:
        load_model(model, pretrained, device)
    print(model)

    return model

def load_model(model, pretrained, device):
    weights = torch.load(pretrained, map_location=device)
    model.load_state_dict(weights['model'], strict=False)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg", ".png", ".jpeg", ".bmp"])

def str_to_list(x):
    return [int(xi) for xi in x.split(',')]

def record_scalar(writer, scalar_list, scalar_name_list, cur_iter):
    scalar_name_list = scalar_name_list[1:-1].split(',')
    for idx, item in enumerate(scalar_list):
        writer.add_scalar(scalar_name_list[idx].strip(' '), item, cur_iter)


def record_image(writer, image_list, cur_iter, num_rows=8):
    image_to_show = torch.cat(image_list, dim=0)
    writer.add_image('visualization', make_grid(image_to_show, nrow=num_rows), cur_iter)


def save_checkpoint(name, model, epoch, iteration, prefix=""):
    model_out_path = "./saves/{}/".format(name) + prefix + "model_epoch_{}_iter_{}.pth".format(epoch, iteration)
    state = {"epoch": epoch, "model": model.state_dict()}
    if not os.path.exists("./saves/{}/".format(name)):
        os.makedirs("./saves/{}/".format(name))

    torch.save(state, model_out_path)

    print("model checkpoint saved @ {}".format(model_out_path))

def calc_reconstruction_loss(x, recon_x, loss_type='mse', reduction='sum'):
    """

    :param x: original inputs
    :param recon_x:  reconstruction of the VAE's input
    :param loss_type: "mse", "l1", "bce"
    :param reduction: "sum", "mean", "none"
    :return: recon_loss
    """
    if reduction not in ['sum', 'mean', 'none']:
        raise NotImplementedError
    recon_x = recon_x.view(recon_x.size(0), -1)
    x = x.view(x.size(0), -1)
    if loss_type == 'mse':
        recon_error = F.mse_loss(recon_x, x, reduction='none')
        recon_error = recon_error.sum(1)
        if reduction == 'sum':
            recon_error = recon_error.sum()
        elif reduction == 'mean':
            recon_error = recon_error.mean()
    elif loss_type == 'l1':
        recon_error = F.l1_loss(recon_x, x, reduction=reduction)
    elif loss_type == 'bce':
        recon_error = F.binary_cross_entropy(recon_x, x, reduction=reduction)
    else:
        raise NotImplementedError
    return recon_error

def calc_kl(logvar, mu, mu_o=0.0, logvar_o=0.0, reduce='sum'):
    """
    Calculate kl-divergence
    :param logvar: log-variance from the encoder
    :param mu: mean from the encoder
    :param mu_o: negative mean for outliers (hyper-parameter)
    :param logvar_o: negative log-variance for outliers (hyper-parameter)
    :param reduce: type of reduce: 'sum', 'none'
    :return: kld
    """
    if not isinstance(mu_o, torch.Tensor):
        mu_o = torch.tensor(mu_o).to(mu.device)
    if not isinstance(logvar_o, torch.Tensor):
        logvar_o = torch.tensor(logvar_o).to(mu.device)
    kl = -0.5 * (1 + logvar - logvar_o - logvar.exp() / torch.exp(logvar_o) - (mu - mu_o).pow(2) / torch.exp(
        logvar_o)).sum(1)
    if reduce == 'sum':
        kl = torch.sum(kl)
    elif reduce == 'mean':
        kl = torch.mean(kl)
    return kl


def get_coef(iter_,epoch_iter,epoch,mode='linear'):
    total = epoch_iter*epoch
    if mode == 'linear':
        scaled_iter_ = iter_*5/total
    if mode == 'tanh':
        scaled_iter_ = math.tanh(iter_*5/total)
    return scaled_iter_ if scaled_iter_ <= 1 else 1

def reduce_with_choice(mu1, mu2, var1, var2, choice = None, eta = 0):
    term1 = torch.pow(mu1-mu2,2)
    term2 = torch.div(term1,eta+var1+var2)
    term3 = torch.mul(term2,-0.5)
    term4 = torch.exp(term3)
    term5 = torch.sqrt(var1+var2) + eta
    res = torch.div(term4, term5)
    return torch.mean(res) if choice == 'mean' else torch.sum(res)


class Gaussian_Distance(nn.Module):
    def __init__(self,kern=1):
        super(Gaussian_Distance, self).__init__()
        self.kern=kern
        self.avgpool = nn.AvgPool2d(kernel_size=kern, stride=kern)


    def forward(self, mu_a,logvar_a,mu_b,logvar_b,reduce='mean'):

        var_a = torch.exp(logvar_a)
        var_b = torch.exp(logvar_b)

        mu_a1 = mu_a.view(mu_a.size(0),1,-1)
        mu_a2 = mu_a.view(1,mu_a.size(0),-1)
        var_a1 = var_a.view(var_a.size(0),1,-1)
        var_a2 = var_a.view(1,var_a.size(0),-1)

        mu_b1 = mu_b.view(mu_b.size(0),1,-1)
        mu_b2 = mu_b.view(1,mu_b.size(0),-1)
        var_b1 = var_b.view(var_b.size(0),1,-1)
        var_b2 = var_b.view(1,var_b.size(0),-1)

        if reduce == 'mean':
            vaa = reduce_with_choice(mu_a1, mu_a2, var_a1, var_a2, choice = 'mean')
            vab = reduce_with_choice(mu_a1, mu_b2, var_a1, var_b2, choice = 'mean')
            vbb = reduce_with_choice(mu_b1, mu_b2, var_b1, var_b2, choice = 'mean')
        
        else:
            vaa = reduce_with_choice(mu_a1, mu_a2, var_a1, var_a2, choice = 'sum')
            vab = reduce_with_choice(mu_a1, mu_b2, var_a1, var_b2, choice = 'sum')
            vbb = reduce_with_choice(mu_b1, mu_b2, var_b1, var_b2, choice = 'sum')


        loss = vaa+vbb-torch.mul(vab,2.0)

        return loss





        # NOTE: comment this fid score
        # if with_fid and ((epoch == 0) or (epoch >= 100 and epoch % 20 == 0) or epoch == num_epochs - 1):
        #     with torch.no_grad():
        #         print("calculating fid...")
        #         fid = calculate_fid_given_dataset(train_data_loader, model, batch_size, cuda=True, dims=2048,
        #                                           device=device, num_images=50000)
        #         print("fid:", fid)
        #         if best_fid is None:
        #             best_fid = fid
        #         elif best_fid > fid:
        #             print("best fid updated: {} -> {}".format(best_fid, fid))
        #             best_fid = fid
        #             # save
        #             save_epoch = epoch
        #             prefix = dataset + "g_intro" + "_betas_" + str(beta_kl) + "_" + str(beta_neg) + "_" + str(
        #                 beta_rec) + "_" + "fid_" + str(fid) + "_"
        #             save_checkpoint(name, model, save_epoch, cur_iter, prefix)


class EMA():
    '''
    # Exponential Moving Average
    # Here is a simple usage
    ema = EMA(model, 0.999)
    ema.register()

    def train():
        optimizer.step()
        ema.update()

    def evaluate():
        ema.apply_shadow()
        # evaluate
        ema.restore()
    '''
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        # param -> shadow
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        # moving average (param & shadow) -> shadow
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        # param -> backup
        # shadow -> param
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        # backup -> param
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}