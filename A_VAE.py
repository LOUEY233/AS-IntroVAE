"""
Train G-Intro VAE for image datasets
Author: Changjie Lu
"""
# Purpose: 
# linear decay (line:165)

# imports
# torch and friends
from functools import reduce
from tkinter import Variable
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST, SVHN, LSUN
from torchvision import transforms

import os
import random
import time
import numpy as np
from tqdm import tqdm
import pickle
# from dataset import ImageDatasetFromFile, DigitalMonstersDataset
from metrics.fid_score import calculate_fid_given_dataset
import matplotlib.pyplot as plt
import matplotlib
from utils import *

matplotlib.use('Agg')

from utils import *


"""
Train Functions
"""

def Train_AVAE(dataset='cifar10', z_dim=128, lr_d=2e-4, lr_g=2e-4, batch_size=128, num_workers=4,
                start_epoch=0, exit_on_negative_diff=False,
                num_epochs=250, num_vae=0, save_interval=25, recon_loss_type="mse",
                beta_kl=1.0, beta_rec=1.0, beta_neg=1.0, test_iter=1000, seed=-1, pretrained=None,
                device=torch.device("cpu"), num_row=8, gamma_r=1e-8, with_fid=True,device_ids=[0]):
    """
    :param dataset: dataset to train on: ['cifar10', 'mnist', 'fmnist', 'svhn', 'monsters128', 'celeb128', 'celeb256', 'celeb1024']
    :param z_dim: latent dimensions
    :param lr_d: learning rate for discriminator
    :param lr_g: learning rate for generator
    :param batch_size: batch size
    :param num_workers: num workers for the loading the data
    :param start_epoch: epoch to start from
    :param exit_on_negative_diff: stop run if mean kl diff between fake and real is negative after 50 epochs
    :param num_epochs: total number of epochs to run
    :param num_vae: number of epochs for vanilla vae training
    :param save_interval: epochs between checkpoint saving
    :param recon_loss_type: type of reconstruction loss ('mse', 'l1', 'bce')
    :param beta_kl: beta coefficient for the kl divergence
    :param beta_rec: beta coefficient for the reconstruction loss
    :param beta_neg: beta coefficient for the kl divergence in the expELBO function
    :param test_iter: iterations between sample image saving
    :param seed: seed
    :param pretrained: path to pretrained model, to continue training
    :param device: device to run calculation on - torch.device('cuda:x') or torch.device('cpu')
    :param num_row: number of images in a row gor the sample image saving
    :param gamma_r: coefficient for the reconstruction loss for fake data in the decoder
    :param with_fid: calculate FID during training (True/False)
    :return:
    """
    # Set random seed
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        print("random seed: ", seed)
    
    # Set Result Path and Model Name
    fig_dir = './results/AVAE5/figures_' + dataset
    name = 'G_VAE5' 
    os.makedirs(fig_dir, exist_ok=True)

    # Get Dataset
    # NOTE: ch for pictures, usually 3 or 1
    # NOTE: channels for latent space
    ch, channels,image_size,train_data_loader = get_dataset(dataset,batch_size,num_workers)

    # Build Model
    model = get_model(z_dim,pretrained,ch, channels, image_size,device)
    model.encoder = nn.DataParallel(model.encoder,device_ids=device_ids)
    model.decoder = nn.DataParallel(model.decoder,device_ids=device_ids)

    # Build Exponential Moving Average
    ema = EMA(model.decoder,0.999)
    ema.register()

    # Build Optimizer
    optimizer_d = optim.Adam(model.encoder.parameters(), lr=lr_d)
    optimizer_g = optim.Adam(model.decoder.parameters(), lr=lr_g)

    # Build Learning Rate Scheduler
    e_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_d, milestones=(350,), gamma=0.1)
    d_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_g, milestones=(350,), gamma=0.1)

    # Calculate Gaussian Distance
    distance = Gaussian_Distance(1)

    # Normalize 
    scale = 1 / (ch * image_size ** 2)  # normalize by images size (channels * height * width)
    start_time = time.time()

    cur_iter = 0
    kls_real = []
    kls_fake = []
    kls_rec = []
    rec_errs = []
    exp_elbos_f = []
    exp_elbos_r = []
    # best_fid = None

    # Per Epoch Training
    for epoch in range(start_epoch, num_epochs):
        # NOTE: (see util.py for calculating fid score)

        # save models
        if epoch % save_interval == 0 and epoch > 0:
            save_epoch = (epoch // save_interval) * save_interval
            prefix = dataset + "_g_intro" + "_betas_" + str(beta_kl) + "_" + str(beta_neg) + "_" + str(
                beta_rec) + "_"
            save_checkpoint(name, model, save_epoch, cur_iter, prefix)

        model.train()

        diff_kls = []
        batch_kls_real = []
        batch_kls_fake = []
        batch_kls_rec = []
        batch_rec_errs = []
        batch_exp_elbo_f = []
        batch_exp_elbo_r = []

        pbar = tqdm(iterable=train_data_loader)
        total_iter = len(pbar)

        # Per Iteration Training
        for batch in pbar:


            # --------------train------------
            # TODO: add comments, simplify code
            if dataset in ["cifar10", "svhn", "fmnist", "mnist",'lsun']:
                batch = batch[0]
            c = get_coef(cur_iter,epoch_iter=total_iter,epoch=num_epochs,mode='tanh')
            print(c)
            if len(batch.size()) == 3:
                batch = batch.unsqueeze(0)

            b_size = batch.size(0)
            noise_batch = torch.randn(size=(b_size, z_dim)).to(device)

            real_batch = batch.to(device)

            # =========== Update E ================
            for param in model.encoder.parameters():
                param.requires_grad = True
            for param in model.decoder.parameters():
                param.requires_grad = False

            fake = model.sample(noise_batch)

            real_mu, real_logvar = model.encode(real_batch)
            z = reparameterize(real_mu, real_logvar)
            rec = model.decoder(z)

            loss_rec = calc_reconstruction_loss(real_batch, rec, loss_type=recon_loss_type, reduction="mean")

            lossE_real_kl = calc_kl(real_logvar, real_mu, reduce="mean")

            rec_mu, rec_logvar, z_rec, rec_rec = model(rec.detach())
            fake_mu, fake_logvar, z_fake, rec_fake = model(fake.detach())

            kl_rec = calc_kl(rec_logvar, rec_mu, reduce="none")
            kl_fake = calc_kl(fake_logvar, fake_mu, reduce="none")

            loss_rec_rec_e = calc_reconstruction_loss(rec, rec_rec, loss_type=recon_loss_type, reduction='none')
            while len(loss_rec_rec_e.shape) > 1:
                loss_rec_rec_e = loss_rec_rec_e.sum(-1)
            loss_rec_fake_e = calc_reconstruction_loss(fake, rec_fake, loss_type=recon_loss_type, reduction='none')
            while len(loss_rec_fake_e.shape) > 1:
                loss_rec_fake_e = loss_rec_fake_e.sum(-1)
            
            rec_mu = nn.Tanh()(rec_mu)
            rec_logvar = nn.Tanh()(rec_logvar)
            fake_mu = nn.Tanh()(fake_mu)
            fake_logvar = nn.Tanh()(fake_logvar)

            dis = distance(rec_mu,rec_logvar,fake_mu,fake_logvar,reduce='none')

            # NOTE: depends on performance, may change these two lines with backup
            expelbo_rec = (-2 * scale * (beta_rec * loss_rec_rec_e + beta_neg * (c*kl_rec+(1-c)*dis))).exp().mean()
            expelbo_fake = (-2 * scale * (beta_rec * loss_rec_fake_e + beta_neg * (c*kl_fake+(1-c)*dis))).exp().mean()

            lossE_fake = 0.25 * (expelbo_rec + expelbo_fake)
            lossE_real = scale * (beta_rec * loss_rec + beta_kl * lossE_real_kl)

            lossE = lossE_real + lossE_fake
            if cur_iter % 50 == 0:
                print("lossE fake:{},lossE real:{}".format(scale *beta_neg * kl_rec,lossE_real))
                print('coef',c)
            optimizer_d.zero_grad()
            lossE.backward()
            optimizer_d.step()


            # ========= Update D ==================
            for param in model.encoder.parameters():
                param.requires_grad = False
            for param in model.decoder.parameters():
                param.requires_grad = True

            fake = model.sample(noise_batch)
            rec = model.decoder(z.detach())
            loss_rec = calc_reconstruction_loss(real_batch, rec, loss_type=recon_loss_type, reduction="mean")

            rec_mu, rec_logvar = model.encode(rec)
            z_rec = reparameterize(rec_mu, rec_logvar)

            fake_mu, fake_logvar = model.encode(fake)
            z_fake = reparameterize(fake_mu, fake_logvar)

            rec_rec = model.decode(z_rec.detach())
            rec_fake = model.decode(z_fake.detach())

            loss_rec_rec = calc_reconstruction_loss(rec.detach(), rec_rec, loss_type=recon_loss_type,
                                                    reduction="mean")
            loss_fake_rec = calc_reconstruction_loss(fake.detach(), rec_fake, loss_type=recon_loss_type,
                                                        reduction="mean")
                                                        
            rec_mu = nn.Tanh()(rec_mu)
            rec_logvar = nn.Tanh()(rec_logvar)
            fake_mu = nn.Tanh()(fake_mu)
            fake_logvar = nn.Tanh()(fake_logvar)

            lossD_rec_kl = calc_kl(rec_logvar, rec_mu, reduce="mean")
            lossD_fake_kl = calc_kl(fake_logvar, fake_mu, reduce="mean")
            dis = distance(rec_mu,rec_logvar,fake_mu,fake_logvar,reduce='mean')

            lossD = scale * (loss_rec * beta_rec + (
                    c*(lossD_rec_kl + lossD_fake_kl)+(1-c)*dis) * 0.5 * beta_kl + gamma_r * 0.5 * beta_rec * (
                                        loss_rec_rec + loss_fake_rec))
            if cur_iter % 50 == 0:
                print("lossD:{},lossD kl:{}".format(lossD,lossD_fake_kl+lossD_rec_kl))
            optimizer_g.zero_grad()
            lossD.backward()
            optimizer_g.step()

            ema.update()

            if torch.isnan(lossD) or torch.isnan(lossE):
                raise SystemError

            dif_kl = -lossE_real_kl.data.cpu() + lossD_fake_kl.data.cpu()
            pbar.set_description_str('epoch #{}'.format(epoch))
            pbar.set_postfix(r_loss=loss_rec.data.cpu().item(), kl=lossE_real_kl.data.cpu().item(),
                                diff_kl=dif_kl.item(), expelbo_f=expelbo_fake.cpu().item())

            diff_kls.append(-lossE_real_kl.data.cpu().item() + lossD_fake_kl.data.cpu().item())
            batch_kls_real.append(lossE_real_kl.data.cpu().item())
            batch_kls_fake.append(lossD_fake_kl.cpu().item())
            batch_kls_rec.append(lossD_rec_kl.data.cpu().item())
            batch_rec_errs.append(loss_rec.data.cpu().item())
            batch_exp_elbo_f.append(expelbo_fake.data.cpu())
            batch_exp_elbo_r.append(expelbo_rec.data.cpu())

            if cur_iter % test_iter == 0:
                _, _, _, rec_det = model(real_batch, deterministic=True)
                ema.apply_shadow()
                fake = model.sample(noise_batch)
                max_imgs = min(batch.size(0), 16)
                vutils.save_image(
                    torch.cat([real_batch[:max_imgs], rec_det[:max_imgs], fake[:max_imgs]], dim=0).data.cpu(),
                    '{}/image_{}.jpg'.format(fig_dir, cur_iter), nrow=num_row)
                ema.restore()

            cur_iter += 1
        e_scheduler.step()
        d_scheduler.step()
        pbar.close()
        if exit_on_negative_diff and epoch > 50 and np.mean(diff_kls) < -1.0:
            print(
                f'the kl difference [{np.mean(diff_kls):.3f}] between fake and real is negative (no sampling improvement)')
            print("try to lower beta_neg hyperparameter")
            print("exiting...")
            raise SystemError("Negative KL Difference")

        if epoch > num_vae - 1:
            kls_real.append(np.mean(batch_kls_real))
            kls_fake.append(np.mean(batch_kls_fake))
            kls_rec.append(np.mean(batch_kls_rec))
            rec_errs.append(np.mean(batch_rec_errs))
            exp_elbos_f.append(np.mean(batch_exp_elbo_f))
            exp_elbos_r.append(np.mean(batch_exp_elbo_r))
            # epoch summary
            print('#' * 50)
            print(f'Epoch {epoch} Summary:')
            print(f'beta_rec: {beta_rec}, beta_kl: {beta_kl}, beta_neg: {beta_neg}')
            print(
                f'rec: {rec_errs[-1]:.3f}, kl: {kls_real[-1]:.3f}, kl_fake: {kls_fake[-1]:.3f}, kl_rec: {kls_rec[-1]:.3f}')
            print(
                f'diff_kl: {np.mean(diff_kls):.3f}, exp_elbo_f: {exp_elbos_f[-1]:.4e}, exp_elbo_r: {exp_elbos_r[-1]:.4e}')
            print(f'time: {time.time() - start_time}')
            print('#' * 50)
        if epoch == num_epochs - 1:
            with torch.no_grad():
                _, _, _, rec_det = model(real_batch, deterministic=True)
                noise_batch = torch.randn(size=(b_size, z_dim)).to(device)
                fake = model.sample(noise_batch)
                max_imgs = min(batch.size(0), 16)
                vutils.save_image(
                    torch.cat([real_batch[:max_imgs], rec_det[:max_imgs], fake[:max_imgs]], dim=0).data.cpu(),
                    '{}/image_{}.jpg'.format(fig_dir, cur_iter), nrow=num_row)

            # plot graphs
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(np.arange(len(kls_real)), kls_real, label="kl_real")
            ax.plot(np.arange(len(kls_fake)), kls_fake, label="kl_fake")
            ax.plot(np.arange(len(kls_rec)), kls_rec, label="kl_rec")
            ax.plot(np.arange(len(rec_errs)), rec_errs, label="rec_err")
            ax.legend()
            plt.savefig('./plot_loss/{}.jpg'.format(name))
            with open('./{}.pickle'.format(name), 'wb') as fp:
                graph_dict = {"kl_real": kls_real, "kl_fake": kls_fake, "kl_rec": kls_rec, "rec_err": rec_errs}
                pickle.dump(graph_dict, fp)
            # save models
            prefix = dataset + name + "_betas_" + str(beta_kl) + "_" + str(beta_neg) + "_" + str(
                beta_rec) + "_"
            save_checkpoint(name, model, epoch, cur_iter, prefix)
            model.train()



if __name__ == '__main__':
    """
    Recommended hyper-parameters:
    - CIFAR10: beta_kl: 1.0, beta_rec: 1.0, beta_neg: 256, z_dim: 128, batch_size: 32
    - SVHN: beta_kl: 1.0, beta_rec: 1.0, beta_neg: 256, z_dim: 128, batch_size: 32
    - MNIST: beta_kl: 1.0, beta_rec: 1.0, beta_neg: 256, z_dim: 32, batch_size: 128
    - FashionMNIST: beta_kl: 1.0, beta_rec: 1.0, beta_neg: 256, z_dim: 32, batch_size: 128
    - Monsters: beta_kl: 0.2, beta_rec: 0.2, beta_neg: 256, z_dim: 128, batch_size: 16
    - CelebA-HQ: beta_kl: 1.0, beta_rec: 0.5, beta_neg: 1024, z_dim: 256, batch_size: 8
    """
    beta_kl = 1.0
    beta_rec = 1.0
    beta_neg = 256
    if torch.cuda.is_available():
        torch.cuda.current_device()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print("betas: ", beta_kl, beta_neg, beta_rec)
    try:
        Train_AVAE(dataset="monsters128", z_dim=128, batch_size=16, num_workers=0, num_epochs=400,
                             num_vae=0, beta_kl=beta_kl, beta_neg=beta_neg, beta_rec=beta_rec,
                             device=device, save_interval=50, start_epoch=0, lr_e=2e-4, lr_d=2e-4,
                             pretrained=None,
                             test_iter=1000, with_fid=False)
    except SystemError:
        print("Error, probably loss is NaN, try again...")

