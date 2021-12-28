import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data

import datasets
import models
import models.cub_caption
import models.cub_image
import options
import utils

opt = options.parser.parse_args()
print(opt)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

paired_dataloader_train = torch.utils.data.DataLoader(datasets.CaptionImagePair(
    datasets.CUBCaptionFeature(opt.dataroot, split='train', normalization=None),
    datasets.CUBImageFeature(opt.dataroot, split='train', normalization=None)),
    batch_size=256,
    num_workers=4,
    shuffle=False,
    drop_last=False,
    pin_memory=True)
cap_train = []
img_train = []
for b in paired_dataloader_train:
    c, i = b
    cap_train.append(c)
    img_train.append(i)

cap_train = torch.cat(cap_train, dim=0)
img_train = torch.cat(img_train, dim=0)

cap_mean = cap_train.mean(dim=0, keepdim=True)
img_mean = img_train.mean(dim=0, keepdim=True)

corr, (cap_proj, img_proj) = utils.cca([cap_train, img_train], k=40)

groundtruth = F.cosine_similarity((cap_train - cap_mean) @ cap_proj, (img_train - img_mean) @ img_proj).mean()
print('train set groundtruth: {:.6f}'.format(groundtruth.item()))

cap_mean = cap_mean.to(device)
img_mean = img_mean.to(device)
cap_proj = cap_proj.to(device)
img_proj = img_proj.to(device)

paired_dataloader_test = torch.utils.data.DataLoader(datasets.CaptionImagePair(
    datasets.CUBCaptionFeature(opt.dataroot, split='test', normalization=None),
    datasets.CUBImageFeature(opt.dataroot, split='test', normalization=None)),
    batch_size=256,
    num_workers=4,
    shuffle=False,
    drop_last=False,
    pin_memory=True)

score_gt = []
for x1, x2 in paired_dataloader_test:
    x1 = x1.to(device)
    x2 = x2.to(device)

    score_gt.append(F.cosine_similarity((x1 - cap_mean) @ cap_proj, (x2 - img_mean) @ img_proj).mean())

groundtruth = sum(score_gt) / len(score_gt)
print('test set groundtruth: {:.6f}'.format(groundtruth.item()))


def calc_cross_coherence(loader, cap_encoder, cap_decoder, img_encoder, img_decoder, save_samples_to=None):
    score_gt = []
    score_c2i = []
    score_i2c = []

    cap2img = []
    img2cap = []

    for x1, x2 in loader:
        x1 = x1.to(device)
        x2 = x2.to(device)

        dec_x2 = img_decoder(cap_encoder(x1))
        dec_x1 = cap_decoder(img_encoder(x2))

        cap2img.append(dec_x2)
        img2cap.append(dec_x1)

        score_gt.append(F.cosine_similarity((x1 - cap_mean) @ cap_proj, (x2 - img_mean) @ img_proj).mean())
        score_c2i.append(F.cosine_similarity((x1 - cap_mean) @ cap_proj, (dec_x2 - img_mean) @ img_proj).mean())
        score_i2c.append(F.cosine_similarity((dec_x1 - cap_mean) @ cap_proj, (x2 - img_mean) @ img_proj).mean())

    if save_samples_to:
        cap2img = torch.cat(cap2img, dim=0)
        img2cap = torch.cat(img2cap, dim=0)
        np.save('{}_cap2img.npy'.format(save_samples_to), cap2img.cpu().numpy())
        np.save('{}_img2cap.npy'.format(save_samples_to), img2cap.cpu().numpy())

    return sum(score_gt) / len(score_gt), sum(score_c2i) / len(score_c2i), sum(score_i2c) / len(score_i2c)


def calc_joint_coherence(cap_decoder, img_decoder, total=10000, save_samples_to=None):
    z = torch.randn(total, opt.latent_dim).to(device)

    cap = cap_decoder(z)
    img = img_decoder(z)

    score = F.cosine_similarity((cap - cap_mean) @ cap_proj, (img - img_mean) @ img_proj).mean()
    if save_samples_to:
        np.save('{}_joint_cap.npy'.format(save_samples_to), cap.cpu().numpy())
        np.save('{}_joint_img.npy'.format(save_samples_to), img.cpu().numpy())
    return score


def calc_synergy_coherence(loader, cap_encoder, cap_decoder, img_encoder, img_decoder, save_samples_to=None):
    score_gt = []
    score_c2c = []
    score_i2i = []

    cap2cap = []
    img2img = []
    for x1, x2 in loader:
        x1 = x1.to(device)
        x2 = x2.to(device)

        z1 = cap_encoder.module(x1)
        z2 = img_encoder.module(x2)
        z_joint = utils.reparameterize(utils.joint_posterior(z1, z2, average=True))
        s1 = utils.reparameterize(z1)[:, :opt.style_dim]
        s2 = utils.reparameterize(z2)[:, :opt.style_dim]
        c = z_joint[:, opt.style_dim:]

        dec_x1 = cap_decoder(torch.cat([s1, c], dim=1))
        dec_x2 = img_decoder(torch.cat([s2, c], dim=1))

        cap2cap.append(dec_x1)
        img2img.append(dec_x2)

        score_gt.append(F.cosine_similarity((x1 - cap_mean) @ cap_proj, (x2 - img_mean) @ img_proj).mean())
        score_c2c.append(F.cosine_similarity((dec_x1 - cap_mean) @ cap_proj, (x2 - img_mean) @ img_proj).mean())
        score_i2i.append(F.cosine_similarity((x1 - cap_mean) @ cap_proj, (dec_x2 - img_mean) @ img_proj).mean())

    if save_samples_to:
        cap2cap = torch.cat(cap2cap, dim=0)
        img2img = torch.cat(img2img, dim=0)
        np.save('{}_cap2cap.npy'.format(save_samples_to), cap2cap.cpu().numpy())
        np.save('{}_img2img.npy'.format(save_samples_to), img2img.cpu().numpy())

    return sum(score_gt) / len(score_gt), sum(score_c2c) / len(score_c2c), sum(score_i2i) / len(score_i2i)


def calc_rec_coherence(loader, cap_encoder, cap_decoder, img_encoder, img_decoder):
    score_gt = []
    score_c2c = []
    score_i2i = []

    for x1, x2 in loader:
        x1 = x1.to(device)
        x2 = x2.to(device)

        dec_x1 = cap_decoder(cap_encoder(x1))
        dec_x2 = img_decoder(img_encoder(x2))

        score_gt.append(F.cosine_similarity((x1 - cap_mean) @ cap_proj, (x2 - img_mean) @ img_proj).mean())
        score_c2c.append(F.cosine_similarity((dec_x1 - cap_mean) @ cap_proj, (x2 - img_mean) @ img_proj).mean())
        score_i2i.append(F.cosine_similarity((x1 - cap_mean) @ cap_proj, (dec_x2 - img_mean) @ img_proj).mean())

    return sum(score_gt) / len(score_gt), sum(score_c2c) / len(score_c2c), sum(score_i2i) / len(score_i2i)
    # return sum(score_gt) / len(score_gt), sum(score_c2c) / len(score_c2c), None
    # return sum(score_gt) / len(score_gt), None, sum(score_i2i) / len(score_i2i)


@torch.no_grad()
def main():
    conditional = models.DeterministicConditional if opt.deterministic else models.GaussianConditional

    cap_encoder = conditional(
        models.cub_caption.EncoderFT(latent_dim=opt.latent_dim if opt.deterministic else 2 * opt.latent_dim))
    # cap_encoder = models.cub_caption.EncoderFT(latent_dim=opt.latent_dim)
    cap_decoder = models.cub_caption.DecoderFT(latent_dim=opt.latent_dim)

    img_encoder = conditional(
        models.cub_image.EncoderFT(latent_dim=opt.latent_dim if opt.deterministic else 2 * opt.latent_dim))
    # img_encoder = models.cub_image.EncoderFT(latent_dim=opt.latent_dim)
    img_decoder = models.cub_image.DecoderFT(latent_dim=opt.latent_dim)

    cap_encoder.to(device)
    cap_decoder.to(device)
    img_encoder.to(device)
    img_decoder.to(device)

    cap_encoder.load_state_dict((torch.load(os.path.join(opt.checkpoint_dir, 'cap_encoder.pt'),
                                            map_location=device)))
    cap_decoder.load_state_dict((torch.load(os.path.join(opt.checkpoint_dir, 'cap_decoder.pt'),
                                            map_location=device)))
    img_encoder.load_state_dict((torch.load(os.path.join(opt.checkpoint_dir, 'img_encoder.pt'),
                                            map_location=device)))
    img_decoder.load_state_dict((torch.load(os.path.join(opt.checkpoint_dir, 'img_decoder.pt'),
                                            map_location=device)))

    cap_encoder.eval()
    cap_decoder.eval()
    img_encoder.eval()
    img_decoder.eval()

    # acc_rec_gt, acc_rec_c2c, acc_i2i = calc_rec_coherence(paired_dataloader_test,
    #                                                       cap_encoder, cap_decoder, img_encoder, img_encoder)
    #                                                       cap_encoder, cap_decoder, None, None)
    #                                                       None, None, img_encoder, img_encoder)
    # print('{:.6f}'.format(acc_rec_gt))
    # print('{:.6f}'.format(acc_rec_c2c))
    # print('{:.6f}'.format(acc_i2i))

    print('{:.6f}'.format(groundtruth))

    with torch.no_grad():
        if not opt.deterministic:
            acc_syn_gt, acc_syn_cap, acc_syn_img = calc_synergy_coherence(paired_dataloader_test,
                                                                          cap_encoder, cap_decoder,
                                                                          img_encoder, img_decoder,
                                                                          save_samples_to='samples')
            print('{:.6f}'.format(acc_syn_gt))
            print('{:.6f}'.format(acc_syn_cap))
            print('{:.6f}'.format(acc_syn_img))

        acc_gt, acc_c2i, acc_i2c = calc_cross_coherence(paired_dataloader_test,
                                                        cap_encoder, cap_decoder,
                                                        img_encoder, img_decoder,
                                                        save_samples_to='samples')

        print('{:.6f}'.format(acc_gt))
        print('{:.6f}'.format(acc_c2i))
        print('{:.6f}'.format(acc_i2c))

        acc_joint = calc_joint_coherence(cap_decoder, img_decoder, save_samples_to='samples')
        print('{:.6f}'.format(acc_joint))


if __name__ == '__main__':
    main()
