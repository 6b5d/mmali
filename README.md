# Multimodal Adversarially Learned Inference with Factorized Discriminators

Implementation of paper Multimodal Adversarially Learned Inference with Factorized Discriminators [arxiv](https://arxiv.org/abs/2112.10384)

## Conda environment

create a new environment and install following packages

```shell
conda install python=3.8 pytorch=1.7 cudatoolkit=10.2 torchvision=0.8 torchaudio=0.7 tensorboard \
              scipy matplotlib nltk=3.6 gensim=3.8 scikit-image=0.18 -c pytorch
```

## MultiMNIST

```shell
python train_mmali_multimnist.py --max_iter 250000 \
                                 --style_dim 10 \
                                 --latent_dim 20 \
                                 --lambda_unimodal 0.1 \
                                 --n_modalities 4 \
                                 --name jsd2_mod4
```

## MNIST_SVHN

```shell
python train_mmali_mnist_svhn.py --max_iter 250000 \
                                 --style_dim 10 \
                                 --latent_dim 20 
                                 --use_all \
                                 --lambda_unimodal 0.1 \
                                 --lambda_x_rec 0.05 \
                                 --lambda_s_rec 0.05 \
                                 --lambda_c_rec 0.05 \
                                 --joint_rec \
                                 --name exp_mnist_svhn
```

dataset will be downloaded to /tmp/data and the results will be saved to /tmp/exps

## CUB

### prepare dataset and pretrained model to extract features
1. Download [CUB dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
2. Download the preprocessed char-CNN-RNN text embeddings for [birds](https://drive.google.com/open?id=1j9do5K1BbghwD6W--XvJmbhj21XEEqjV) (From Joint-GAN)
3. run ``python make_cub_img_ft.py`` to extract image embeddings using ResNet101

### training
```shell
python train_mmali_cap_img.py --max_iter 250000 \
                              --style_dim 32 \
                              --latent_dim 64 
                              --lambda_unimodal 1.0 \
                              --lambda_x_rec 1.0 \
                              --name exp_cap_img
```

Use [pretrained model](https://drive.google.com/open?id=1j9do5K1BbghwD6W--XvJmbhj21XEEqjV) to decode sentence features
