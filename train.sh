#SVAE
# python main.py --dataset mnist --device 0,1 --lr 2e-4 --num_epochs 50 --beta_kl 1.0 --beta_rec 1.0 --beta_neg 256 --z_dim 32 --batch_size 128 --model S_VAE
# python main.py --dataset cifar10 --device 0,1 --lr 2e-4 --num_epochs 100 --beta_kl 1.0 --beta_rec 1.0 --beta_neg 256 --z_dim 128 --batch_size 64 --model S_VAE
# python main.py --dataset lsun --device 0,1 --lr 2e-4 --num_epochs 150 --beta_kl 1.0 --beta_rec 1.0 --beta_neg 256 --z_dim 128 --batch_size 32 --model S_VAE
# python main.py --dataset oxford --device 0,1 --lr 2e-4 --num_epochs 150 --beta_kl 1.0 --beta_rec 1.0 --beta_neg 0.5 --z_dim 128 --batch_size 32 --model S_VAE
# python main.py --dataset celeb128 --device 0,1 --lr 2e-4 --num_epochs 150 --beta_kl 0.5--beta_rec 1.0 --beta_neg 256 --z_dim 128 --batch_size 32 --model S_VAE
# python main.py --dataset celeb256 --device 0,1 --lr 2e-4 --num_epochs 150 --beta_kl 0.5 --beta_rec 1.0 --beta_neg 1024 --z_dim 256 --batch_size 16 --model S_VAE

#IVAE
# python main.py --dataset mnist --device 0,1 --lr 2e-4 --num_epochs 50 --beta_kl 1.0 --beta_rec 0.05 --beta_neg 0.5 --z_dim 32 --batch_size 128 --m_plus 10 --model I_VAE
# python main.py --dataset cifar10 --device 0,1 --lr 2e-4 --num_epochs 100 --beta_kl 1.0 --beta_rec 0.05 --beta_neg 0.5 --z_dim 64 --batch_size 64 --m_plus 10 --model I_VAE
# python main.py --dataset lsun --device 0,1 --lr 2e-4 --num_epochs 150 --beta_kl 1.0 --beta_rec 0.05 --beta_neg 0.5 --z_dim 128 --batch_size 32 --m_plus 100 --model I_VAE
# python main.py --dataset oxford --device 0,1 --lr 2e-4 --num_epochs 150 --beta_kl 1.0 --beta_rec 0.05 --beta_neg 0.5 --z_dim 128 --batch_size 32 --m_plus 100 --model I_VAE
# python main.py --dataset celeb128 --device 0,1 --lr 2e-4 --num_epochs 150 --beta_kl 1.0 --beta_rec 0.05 --beta_neg 0.5 --z_dim 128 --batch_size 32 --m_plus 100 --model I_VAE
# python main.py --dataset celeb256 --device 0,1 --lr 2e-4 --num_epochs 150 --beta_kl 1.0 --beta_rec 0.05 --beta_neg 0.5 --z_dim 256 --batch_size 16 --m_plus 120 --model I_VAE

#Ours
python main.py --dataset mnist --device 3,4,5 --lr 2e-4 --num_epochs 100 --beta_kl 1.0 --beta_rec 1.0 --beta_neg 256 --z_dim 32 --batch_size 128 --model A_VAE > ./out_file/AVAE/mnist.out
python main.py --dataset cifar10 --device 3,4,5 --lr 2e-4 --num_epochs 150 --beta_kl 1.0 --beta_rec 1.0 --beta_neg 256 --z_dim 128 --batch_size 64 --model A_VAE > ./out_file/AVAE/cifar.out
python main.py --dataset celeb128 --device 3,4,5 --lr 2e-4 --num_epochs 150 --beta_kl 0.5 --beta_rec 1.0 --beta_neg 256 --z_dim 256 --batch_size 32 --model A_VAE2 > ./out_file/AVAE/celeba128.out 
python main.py --dataset celeb256 --device 0,1,2,3,4,5 --lr 2e-4 --num_epochs 150 --beta_kl 0.5 --beta_rec 1.0 --beta_neg 1024 --z_dim 256 --batch_size 32 --model A_VAE2 > ./out_file/AVAE/celeba256.out
python main.py --dataset oxford --device 3,4,5 --lr 2e-4 --num_epochs 150 --beta_kl 1.0 --beta_rec 1.0 --beta_neg 256 --z_dim 128 --batch_size 32 --model A_VAE > ./out_file/AVAE/oxford.out