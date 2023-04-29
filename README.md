# AS-IntroVAE
This repository contains the official pytorch implement of the paper: \
"AS-IntroVAE: Adversarial Similarity Distance Makes Robust IntroVAEs"

## Updates:
- 2022.8 Paper accepted by ACML'22
- 2023.2 Upload the code

## About this paper:
[paper](https://arxiv.org/pdf/2206.13903.pdf) 
[slides](https://louey233.github.io/data/AS_IntroVAE_slides.pdf) 
[poster](https://louey233.github.io/data/ACML_poster.pdf)

# Training
Run the following script in terminal
```
sh train.sh
```

## Hyperparameters
```python
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
```


# Contact
Please reach lucha@kean.edu for further questions.\
You can also open an issue (prefered) or a pull request in this Github repository.

## BibTeX
Please cite our paper if you find this repository helpful.
```
@article{lu2022introvae,
  title={AS-IntroVAE: Adversarial Similarity Distance Makes Robust IntroVAE},
  author={Lu, Changjie and Zheng, Shen and Wang, Zirui and Dib, Omar and Gupta, Gaurav},
  journal={arXiv preprint arXiv:2206.13903},
  year={2022}
}
```

# Acknowledgement
This repository is heavily borrowed from [Soft-IntroVAE](https://github.com/taldatech/soft-intro-vae-pytorch). Thanks for sharing!




