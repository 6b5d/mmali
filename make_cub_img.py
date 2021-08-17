import os

import torch.nn as nn
import torch.utils.data
import torchvision

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
            torchvision.datasets.ImageFolder(os.path.join(dataroot, 'cub/{}'.format(split)),
                                             transform=torchvision.transforms.Compose([
                                                 torchvision.transforms.Resize(224),
                                                 torchvision.transforms.ToTensor()
                                             ])), batch_size=256, shuffle=False, pin_memory=True)

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

    torch.save({
        'data': ft_mats[1],
        'min': train_min,
        'max': train_max,
        'mean': train_mean,
        'std': train_std,
    }, os.path.join(dataroot, 'cub/processed/cub-img-test.pt'))

    print(ft_mats[0].size(), ft_mats[1].size(), train_mean.size(), train_std.size())
