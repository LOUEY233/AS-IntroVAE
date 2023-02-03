"""
Main function for arguments parsing
Author: Tal Daniel
"""
# imports
import torch
import argparse
from S_VAE import Train_SVAE
from I_VAE import Train_IVAE
from A_VAE import Train_AVAE
# from A_VAE2 import Train_AVAE2
import os

if __name__ == "__main__":
    """
        Recommended hyper-parameters:
        - CIFAR10: beta_kl: 1.0, beta_rec: 1.0, beta_neg: 256, z_dim: 128, batch_size: 32
        - SVHN: beta_kl: 1.0, beta_rec: 1.0, beta_neg: 256, z_dim: 128, batch_size: 32
        - MNIST: beta_kl: 1.0, beta_rec: 1.0, beta_neg: 256, z_dim: 32, batch_size: 128
        - FashionMNIST: beta_kl: 1.0, beta_rec: 1.0, beta_neg: 256, z_dim: 32, batch_size: 128
        - Monsters: beta_kl: 0.2, beta_rec: 0.2, beta_neg: 256, z_dim: 128, batch_size: 16
        - CelebA-HQ: beta_kl: 1.0, beta_rec: 0.5, beta_neg: 1024, z_dim: 256, batch_size: 8
    """
    parser = argparse.ArgumentParser(description="train Soft-IntroVAE")
    parser.add_argument("-d", "--dataset", type=str,
                        help="dataset to train on: ['cifar10', 'mnist','oxford', 'lsun',  'celeb128','celeb256']")
    parser.add_argument("-n", "--num_epochs", type=int, help="total number of epochs to run", default=25)
    parser.add_argument("-z", "--z_dim", type=int, help="latent dimensions", default=128)
    parser.add_argument("-l", "--lr", type=float, help="learning rate", default=2e-4)
    parser.add_argument("-b", "--batch_size", type=int, help="batch size", default=32)
    parser.add_argument("-v", "--num_vae", type=int, help="number of epochs for vanilla vae training", default=0)
    parser.add_argument("-r", "--beta_rec", type=float, help="beta coefficient for the reconstruction loss",
                        default=1.0)
    parser.add_argument("-k", "--beta_kl", type=float, help="beta coefficient for the kl divergence",
                        default=1.0)
    parser.add_argument("-e", "--beta_neg", type=float,
                        help="beta coefficient for the kl divergence in the expELBO function", default=1.0)
    parser.add_argument("-g", "--gamma_r", type=float,
                        help="coefficient for the reconstruction loss for fake data in the decoder", default=1e-8)
    parser.add_argument("-s", "--seed", type=int, help="seed", default=-1)
    parser.add_argument("-p", "--pretrained", type=str, help="path to pretrained model, to continue training",
                        default="None")
    parser.add_argument("--device", type=str, help="device: 0 and up for specific cuda device",
                        default='0,1') # 0,1 for two GPUs
    parser.add_argument('-f', "--fid",default=False, type=bool, help="if specified, FID will be calculated during training")
    parser.add_argument('--model',default='S_VAE',type=str,help='model comparison',choices=['GAN', 'WGAN_GP',
                         'WGAN_SN', 'VAE','I_VAE','S_VAE','A_VAE','A_VAE2'])
    parser.add_argument('--m_plus',default=120,type=int,help='threshold for kl loss')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_ids = [Id for Id in range(torch.cuda.device_count())]
    pretrained = None if args.pretrained == "None" else args.pretrained

    if args.model == 'GAN':
        pass
    elif args.model == 'WGAN_GP':
        pass
    elif args.model == 'WANG_SN':
        pass
    elif args.model == 'VAE':
        pass
    elif args.model == 'S_VAE':
        Train_SVAE(dataset=args.dataset, z_dim=args.z_dim, batch_size=args.batch_size, num_workers=0,
                num_epochs=args.num_epochs,
                num_vae=args.num_vae, beta_kl=args.beta_kl, beta_neg=args.beta_neg, beta_rec=args.beta_rec,
                device=device, save_interval=50, start_epoch=0, lr_d=args.lr, lr_g=args.lr,
                pretrained=pretrained, seed=args.seed,
                test_iter=1000, with_fid=args.fid,device_ids=device_ids)
    if args.model == 'I_VAE':
        Train_IVAE(dataset=args.dataset, z_dim=args.z_dim, batch_size=args.batch_size, num_workers=0,
                num_epochs=args.num_epochs,
                num_vae=args.num_vae, beta_kl=args.beta_kl, beta_neg=args.beta_neg, beta_rec=args.beta_rec,
                device=device, save_interval=50, start_epoch=0, lr_d=args.lr, lr_g=args.lr,
                pretrained=pretrained, seed=args.seed,
                test_iter=1000, with_fid=args.fid,m_plus=args.m_plus,device_ids=device_ids)
    elif args.model == 'A_VAE':
        Train_AVAE(dataset=args.dataset, z_dim=args.z_dim, batch_size=args.batch_size, num_workers=0,
                num_epochs=args.num_epochs,
                num_vae=args.num_vae, beta_kl=args.beta_kl, beta_neg=args.beta_neg, beta_rec=args.beta_rec,
                device=device, save_interval=50, start_epoch=0, lr_d=args.lr, lr_g=args.lr,
                pretrained=pretrained, seed=args.seed,
                test_iter=1000, with_fid=args.fid,device_ids=device_ids)
    # elif args.model == 'A_VAE2':
    #     Train_AVAE2(dataset=args.dataset, z_dim=args.z_dim, batch_size=args.batch_size, num_workers=0,
    #             num_epochs=args.num_epochs,
    #             num_vae=args.num_vae, beta_kl=args.beta_kl, beta_neg=args.beta_neg, beta_rec=args.beta_rec,
    #             device=device, save_interval=50, start_epoch=0, lr_d=args.lr, lr_g=args.lr,
    #             pretrained=pretrained, seed=args.seed,
    #             test_iter=1000, with_fid=args.fid,device_ids=device_ids)
