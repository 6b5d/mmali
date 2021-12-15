# Multimodal Adversarially Learned Inference with Factorized Discriminators

## environment
```shell
conda install python=3.8 pytorch=1.7 cudatoolkit=10.2 torchvision=0.8 torchaudio=0.7 tensorboard \
              scipy matplotlib nltk=3.6 gensim=3.8 scikit-image=0.18 -c pytorch
```

## run
```shell
python train_mmali_mnist_svhn.py --max_iter 250000 \
                                 --style_dim 10 \
                                 --latent_dim 20 
                                 --use_all \
                                 --lambda_unimodal 0.1 \
                                 --name exp_mnist_svhn
```

## TODO