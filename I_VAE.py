"""
Train ori-Intro VAE for image datasets
Author: Tal Daniel
"""

# imports
# torch and friends
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

# standard
import os
import random
import time
import numpy as np
from tqdm import tqdm
import pickle
from dataset import ImageDatasetFromFile, DigitalMonstersDataset
from metrics.fid_score import calculate_fid_given_dataset
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

from utils import *


"""
Train Functions
"""

def Train_IVAE(dataset='cifar10', z_dim=128, lr_d=2e-4, lr_g=2e-4, batch_size=128, num_workers=4,
                start_epoch=0, exit_on_negative_diff=False,
                num_epochs=250, num_vae=0, save_interval=50, recon_loss_type="mse",
                beta_kl=1.0, beta_rec=1.0, beta_neg=1.0, test_iter=1000, seed=-1, pretrained=None,
                device=torch.device("cpu"), num_row=8, gamma_r=1e-8, with_fid=True,m_plus=100,device_ids=[0]):
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
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        print("random seed: ", seed)

    name = 'I_VAE'
    fig_dir = './results/IVAE/figures_' + dataset
    os.makedirs(fig_dir, exist_ok=True)

    ch, channels,image_size,train_data_loader = get_dataset(dataset,batch_size,num_workers)
    model = get_model(z_dim,pretrained,ch, channels, image_size,device)
    model.encoder = nn.DataParallel(model.encoder,device_ids=device_ids)
    model.decoder = nn.DataParallel(model.decoder,device_ids=device_ids)

    optimizer_d = optim.Adam(model.encoder.parameters(), lr=lr_d)
    optimizer_g = optim.Adam(model.decoder.parameters(), lr=lr_g)

    e_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_d, milestones=(350,), gamma=0.1)
    d_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_g, milestones=(350,), gamma=0.1)

    start_time = time.time()

    cur_iter = 0
    kls_real = []
    kls_fake = []
    kls_rec = []
    rec_errs = []
    best_fid = None
    for epoch in range(start_epoch, num_epochs):
        if with_fid and ((epoch == 0) or (epoch >= 100 and epoch % 20 == 0) or epoch == num_epochs - 1):
            with torch.no_grad():
                print("calculating fid...")
                fid = calculate_fid_given_dataset(train_data_loader, model, batch_size, cuda=True, dims=2048,
                                                  device=device, num_images=50000)
                print("fid:", fid)
                if best_fid is None:
                    best_fid = fid
                elif best_fid > fid:
                    print("best fid updated: {} -> {}".format(best_fid, fid))
                    best_fid = fid
                    # save
                    save_epoch = epoch
                    prefix = dataset + "ori_intro" + "_betas_" + str(beta_kl) + "_" + str(beta_neg) + "_" + str(
                        beta_rec) + "_" + "fid_" + str(fid) + "_"
                    save_checkpoint(name,model, save_epoch, cur_iter, prefix)

        diff_kls = []
        # save models
        if epoch % save_interval == 0 and epoch > 0:
            save_epoch = (epoch // save_interval) * save_interval
            prefix = dataset + "_ori_intro" + "_betas_" + str(beta_kl) + "_" + str(beta_neg) + "_" + str(
                beta_rec) + "_"
            save_checkpoint(name, model, save_epoch, cur_iter, prefix)

        model.train()

        batch_kls_real = []
        batch_kls_fake = []
        batch_kls_rec = []
        batch_rec_errs = []

        pbar = tqdm(iterable=train_data_loader)

        for batch in pbar:
            # --------------train------------
            if dataset in ["cifar10", "svhn", "fmnist", "mnist",'lsun']:
                batch = batch[0]
            if epoch < num_vae:
                if len(batch.size()) == 3:
                    batch = batch.unsqueeze(0)

                batch_size = batch.size(0)

                real_batch = batch.to(device)

                # =========== Update E, D ================

                real_mu, real_logvar, z, rec = model(real_batch)

                loss_rec = calc_reconstruction_loss(real_batch, rec, loss_type=recon_loss_type, reduction="mean")
                loss_kl = calc_kl(real_logvar, real_mu, reduce="mean")

                loss = beta_rec * loss_rec + beta_kl * loss_kl

                optimizer_d.zero_grad()
                optimizer_g.zero_grad()
                loss.backward()
                optimizer_d.step()
                optimizer_g.step()

                pbar.set_description_str('epoch #{}'.format(epoch))
                pbar.set_postfix(r_loss=loss_rec.data.cpu().item(), kl=loss_kl.data.cpu().item())

                print(cur_iter)
                if cur_iter % test_iter == 0:
                    vutils.save_image(torch.cat([real_batch, rec], dim=0).data.cpu(),
                                      '{}/image_{}.jpg'.format(fig_dir, cur_iter), nrow=num_row)

            else:
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
                rec = model.decoder(z)            # real image reconstruction
                rec_mu,rec_logvar = model.encode(rec.detach())
                fake_mu,fake_logvar = model.encode(fake.detach())

# recon loss for real
                loss_rec = calc_reconstruction_loss(real_batch, rec, loss_type=recon_loss_type, reduction="mean")
# latent space divergence
                lossE_real_kl = calc_kl(real_logvar, real_mu, reduce="mean")
                kl_rec = calc_kl(rec_logvar, rec_mu, reduce="mean")
                kl_fake = calc_kl(fake_logvar, fake_mu, reduce="mean")

                loss_margin = lossE_real_kl + \
                              F.relu(m_plus-kl_rec) + \
                              F.relu(m_plus-kl_fake) * 0.5 * beta_neg

                # 256*256
                # beta_rec=0.05, beta_kl=1.0, beta_neg = 0.5
                lossE = loss_rec * beta_rec + loss_margin * beta_kl

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
                fake_mu, fake_logvar = model.encode(fake)

                lossD_rec_kl = calc_kl(rec_logvar, rec_mu, reduce="mean")
                lossD_fake_kl = calc_kl(fake_logvar, fake_mu, reduce="mean")

                lossD = (lossD_rec_kl + lossD_fake_kl) * 0.5 * beta_kl

                optimizer_g.zero_grad()
                lossD.backward()
                optimizer_g.step()
                if torch.isnan(lossD) or torch.isnan(lossE):
                    raise SystemError

                dif_kl = -lossE_real_kl.data.cpu() + lossD_fake_kl.data.cpu()
                pbar.set_description_str('epoch #{}'.format(epoch))
                pbar.set_postfix(r_loss=loss_rec.data.cpu().item(), kl=lossE_real_kl.data.cpu().item(),
                                 diff_kl=dif_kl.item(),)

                diff_kls.append(-lossE_real_kl.data.cpu().item() + lossD_fake_kl.data.cpu().item())
                batch_kls_real.append(lossE_real_kl.data.cpu().item())
                batch_kls_fake.append(lossD_fake_kl.cpu().item())
                batch_kls_rec.append(lossD_rec_kl.data.cpu().item())
                batch_rec_errs.append(loss_rec.data.cpu().item())

                if cur_iter % test_iter == 0:
                    _, _, _, rec_det = model(real_batch, deterministic=True)
                    max_imgs = min(batch.size(0), 16)
                    vutils.save_image(
                        torch.cat([real_batch[:max_imgs], rec_det[:max_imgs], fake[:max_imgs]], dim=0).data.cpu(),
                        '{}/image_{}.jpg'.format(fig_dir, cur_iter), nrow=num_row)

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
            # epoch summary
            print('#' * 50)
            print(f'Epoch {epoch} Summary:')
            print(f'beta_rec: {beta_rec}, beta_kl: {beta_kl}, beta_neg: {beta_neg}')
            print(
                f'rec: {rec_errs[-1]:.3f}, kl: {kls_real[-1]:.3f}, kl_fake: {kls_fake[-1]:.3f}, kl_rec: {kls_rec[-1]:.3f}')
            print(
                f'diff_kl: {np.mean(diff_kls):.3f}')
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
            plt.savefig('./ori_intro_train_graphs.jpg')
            with open('./ori_intro_train_graphs_data.pickle', 'wb') as fp:
                graph_dict = {"kl_real": kls_real, "kl_fake": kls_fake, "kl_rec": kls_rec, "rec_err": rec_errs}
                pickle.dump(graph_dict, fp)
            # save models
            prefix = dataset + "_ori_intro" + "_betas_" + str(beta_kl) + "_" + str(beta_neg) + "_" + str(
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
        Train_IVAE(dataset="monsters128", z_dim=128, batch_size=16, num_workers=0, num_epochs=400,
                             num_vae=0, beta_kl=beta_kl, beta_neg=beta_neg, beta_rec=beta_rec,
                             device=device, save_interval=50, start_epoch=0, lr_e=2e-4, lr_d=2e-4,
                             pretrained=None,
                             test_iter=1000, with_fid=False)
    except SystemError:
        print("Error, probably loss is NaN, try again...")
