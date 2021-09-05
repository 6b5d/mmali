import os
import pickle

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
from PIL import Image


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class CUBImage(data.Dataset):
    def __init__(self, root, split='train', transform=None):
        super().__init__()

        self.root = root
        self.split = split
        self.transform = transform

        with open(os.path.join(self.root, 'cub/processed/{}/class_info.pickle'.format(split)), 'rb') as f:
            cls_info = pickle.load(f, encoding='bytes')
        self.cls_info = cls_info

        with open(os.path.join(self.root, 'cub/processed/{}/filenames.pickle'.format(split)), 'rb') as f:
            filenames = pickle.load(f, encoding='bytes')
            paths = [os.path.join(self.root, 'cub/images/{}.jpg'.format(fn)) for fn in filenames]

        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = pil_loader(self.paths[index])
        if self.transform:
            img = self.transform(img)
        return img, self.cls_info[index]


if __name__ == '__main__':
    dataroot = '/tmp/data'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    resnet101 = torchvision.models.resnet101(pretrained=True)
    modules = list(resnet101.children())[:-1]
    featurizer = nn.Sequential(*modules)

    featurizer.to(device)
    featurizer.eval()
    ft_mats = []
    for split in ['train', 'test']:
        dataloader = torch.utils.data.DataLoader(
            CUBImage(dataroot, split=split, transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor()
            ])), batch_size=512, shuffle=False, pin_memory=True)

        with torch.no_grad():
            ft_mat = torch.cat([featurizer(data[0].to(device)).squeeze() for data in dataloader])

        print(ft_mat.shape)
        ft_mats.append(ft_mat)

    train_min = ft_mats[0].min()
    train_max = ft_mats[0].max()
    train_mean = ft_mats[0].mean()
    train_std = ft_mats[0].std()

    print(train_min.item(), train_max.item())
    os.makedirs(os.path.join(dataroot, 'cub/processed/'), exist_ok=True)
    torch.save({
        'data': ft_mats[0],
        'min': train_min,
        'max': train_max,
        'mean': train_mean,
        'std': train_std,
    }, os.path.join(dataroot, 'cub/processed/cub-img-train.pt'))
    print('saved to {}'.format(os.path.join(dataroot, 'cub/processed/cub/processed/cub-img-train.pt')))

    torch.save({
        'data': ft_mats[1],
        'min': train_min,
        'max': train_max,
        'mean': train_mean,
        'std': train_std,
    }, os.path.join(dataroot, 'cub/processed/cub-img-test.pt'))
    print('saved to {}'.format(os.path.join(dataroot, 'cub/processed/cub/processed/cub-img-test.pt')))

    print(ft_mats[0].size(), ft_mats[1].size(), train_mean.size(), train_std.size())
