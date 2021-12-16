# Multimodal Adversarially Learned Inference with Factorized Discriminators

## Conda environment

create a new environment and install following packages

```shell
conda install python=3.8 pytorch=1.7 cudatoolkit=10.2 torchvision=0.8 torchaudio=0.7 tensorboard \
              scipy matplotlib nltk=3.6 gensim=3.8 scikit-image=0.18 -c pytorch
```

## MultiMNIST

```train_mmali_multimnist.py
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
                                 --name exp_mnist_svhn
```

dataset will be downloaded to /tmp/data and the results will be saved to /tmp/exps

## CUB: TODO

### prepare dataset and pretrained model to extract features