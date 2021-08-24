import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from nltk.tokenize import sent_tokenize, word_tokenize

import datasets
import models
import models.cub_caption
import models.cub_image
import options
import utils
from utils import OrderedCounter

opt = options.parser.parse_args()
print(opt)

min_count = 3
len_window = 3
emb_size = 128
seq_len = 32
epochs = 10
sym_exc = '<exc>'
sym_pad = '<pad>'
sym_eos = '<eos>'
raw_data_train_path = os.path.join(opt.dataroot, 'cub/text_trainvalclasses.txt')
raw_data_test_path = os.path.join(opt.dataroot, 'cub/text_testclasses.txt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_vocab_from(list_of_words):
    w2i = dict()
    i2w = dict()

    special_tokens = [sym_exc, sym_pad, sym_eos]
    for st in special_tokens:
        i2w[len(w2i)] = st
        w2i[st] = len(w2i)

    occ_register = OrderedCounter()
    for words in list_of_words:
        occ_register.update(words)

    for w, occ in occ_register.items():
        if occ > min_count and w not in special_tokens:
            i2w[len(w2i)] = w
            w2i[w] = len(w2i)

    assert len(w2i) == len(i2w)

    vocab = dict(w2i=w2i, i2w=i2w)
    return vocab


def create_data_from(list_of_words, vocab):
    list_of_indexes = []

    for words in list_of_words:
        tok = words[:seq_len - 1]
        tok = tok + [sym_eos]
        length = len(tok)
        if seq_len > length:
            tok.extend([sym_pad] * (seq_len - length))
        idx = [vocab['w2i'].get(w, vocab['w2i'][sym_exc]) for w in tok]

        list_of_indexes.append(idx)

    return list_of_indexes


def apply_weights(word_emb, weights, data):
    def truncate(s):
        # np.where works for pytorch boolean tensor
        return s[:np.where(s == 2)[0][0] + 1] if 2 in s else s

    batch_emb = []
    for sent_i in data:
        sent_trunc = truncate(sent_i)
        emb_stacked = torch.stack([word_emb[idx] for idx in sent_trunc])
        weights_stacked = torch.stack([weights[idx] for idx in sent_trunc])
        batch_emb.append(torch.sum(emb_stacked * weights_stacked.unsqueeze(-1), dim=0) / emb_stacked.shape[0])

    return torch.stack(batch_emb, dim=0)


def apply_pc(sent_emb, u):
    return torch.cat([e - torch.matmul(u, e.unsqueeze(-1)).squeeze() for e in sent_emb.split(2048, 0)])


def word2index(list_of_words, vocab):
    list_of_indexes = []
    for words in list_of_words:
        list_of_indexes.append([vocab['w2i'].get(w, vocab['w2i'][sym_exc]) for w in words])
    return torch.from_numpy(np.array(list_of_indexes)).int()


# see also https://openreview.net/forum?id=SyK00v5xx
def index2emb(x, embeddings, weights, u):
    return apply_pc(apply_weights(embeddings, weights, x), u)


def calculate_corr(captions, images, cap_mean, cap_proj, img_mean, img_proj):
    return F.cosine_similarity((captions - cap_mean) @ cap_proj, (images - img_mean) @ img_proj).mean()


with open(os.path.join(opt.dataroot, 'CCA_emb/cub.emb'), 'rb') as file:
    embeddings = pickle.load(file)
    embeddings = torch.from_numpy(embeddings).float()
    print('loaded emb')

with open(os.path.join(opt.dataroot, 'CCA_emb/cub.weights'), 'rb') as file:
    weights = pickle.load(file)
    weights = torch.from_numpy(weights).float()
    print('loaded weights')

with open(os.path.join(opt.dataroot, 'CCA_emb/cub.pc'), 'rb') as file:
    u = pickle.load(file)
    u = u.float()
    print('loaded u')

with open(raw_data_train_path, 'r') as file:
    sentences_train = sent_tokenize(file.read())
list_of_words_train = [word_tokenize(line) for line in sentences_train]

with open(raw_data_test_path, 'r') as file:
    sentences_test = sent_tokenize(file.read())
list_of_words_test = [word_tokenize(line) for line in sentences_test]
vocabulary = create_vocab_from(list_of_words_train)
data_test = torch.from_numpy(np.array(create_data_from(list_of_words_test, vocabulary))).int()

x1_dataset = datasets.CUBCaptionVector(opt.dataroot, split='test', normalization='min-max')
x2_dataset = datasets.CUBImageFeature(opt.dataroot, split='test', normalization='min-max')
paired_dataset = datasets.CaptionImagePair(x1_dataset, x2_dataset)
paired_dataloader = torch.utils.data.DataLoader(paired_dataset,
                                                batch_size=opt.batch_size,
                                                num_workers=opt.n_cpu,
                                                shuffle=False,
                                                drop_last=False,
                                                pin_memory=True)

cap_emb = index2emb(data_test, embeddings, weights, u)
img_emb = torch.repeat_interleave(x2_dataset.decode(x2_dataset.data), 10, dim=0)
cap_mean = cap_emb.mean(dim=0, keepdim=True)
img_mean = img_emb.mean(dim=0, keepdim=True)

corr, (cap_proj, img_proj) = utils.cca([cap_emb, img_emb], k=40)

groundtruth = calculate_corr(cap_emb, img_emb, cap_mean, cap_proj, img_mean, img_proj)
print('Largest eigen value from CCA: {:.3f}'.format(corr[0]))
print('gt corr:', groundtruth)

embeddings = embeddings.to(device)
weights = weights.to(device)
u = u.to(device)
cap_mean = cap_mean.to(device)
cap_proj = cap_proj.to(device)
img_mean = img_mean.to(device)
img_proj = img_proj.to(device)


def calc_cross_coherence(loader, cap_encoder, cap_decoder, img_encoder, img_decoder):
    cap = []
    img = []
    cap2img = []
    img2cap = []

    for x1, x2 in loader:
        x1 = x1.to(device)
        x2 = x2.to(device)

        dec_x2 = img_decoder(cap_encoder(x1))
        dec_x1 = cap_decoder(img_encoder(x2))

        cap.append(x1)
        img.append(x2)
        img2cap.append(dec_x1)
        cap2img.append(dec_x2)

    cap = index2emb(word2index(x1_dataset.decode(torch.cat(cap, dim=0)), vocabulary), embeddings, weights, u)
    img = x2_dataset.decode(torch.cat(img, dim=0))
    cap2img = x2_dataset.decode(torch.cat(cap2img, dim=0))
    img2cap = index2emb(word2index(x1_dataset.decode(torch.cat(img2cap, dim=0)), vocabulary), embeddings, weights, u)

    score_cap2img = calculate_corr(cap, cap2img, cap_mean, cap_proj, img_mean, img_proj)
    score_img2cap = calculate_corr(img2cap, img, cap_mean, cap_proj, img_mean, img_proj)
    return score_cap2img, score_img2cap


def calc_joint_coherence(cap_decoder, img_decoder, total=10000):
    z = torch.randn(total, opt.latent_dim).to(device)

    cap = cap_decoder(z)
    img = img_decoder(z)

    cap = index2emb(word2index(x1_dataset.decode(cap), vocabulary), embeddings, weights, u)
    img = x2_dataset.decode(img)

    score = calculate_corr(cap, img, cap_mean, cap_proj, img_mean, img_proj)
    return score


def calc_synergy_coherence(loader, cap_encoder, cap_decoder, img_encoder, img_decoder):
    cap = []
    img = []
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

        cap.append(x1)
        img.append(x2)
        cap2cap.append(dec_x1)
        img2img.append(dec_x2)

    cap = index2emb(word2index(x1_dataset.decode(torch.cat(cap, dim=0)), vocabulary), embeddings, weights, u)
    img = x2_dataset.decode(torch.cat(img, dim=0))
    img2img = x2_dataset.decode(torch.cat(img2img, dim=0))
    cap2cap = index2emb(word2index(x1_dataset.decode(torch.cat(cap2cap, dim=0)), vocabulary), embeddings, weights, u)

    score_cap2cap = calculate_corr(cap2cap, img, cap_mean, cap_proj, img_mean, img_proj)
    score_img2img = calculate_corr(cap, img2img, cap_mean, cap_proj, img_mean, img_proj)
    return score_cap2cap, score_img2img


@torch.no_grad()
def main():
    conditional = models.DeterministicConditional if opt.deterministic else models.GaussianConditional

    cap_encoder = conditional(
        models.cub_caption.Encoder(latent_dim=opt.latent_dim if opt.deterministic else 2 * opt.latent_dim,
                                   emb_size=opt.emb_size))
    cap_decoder = models.cub_caption.Decoder(latent_dim=opt.latent_dim, emb_size=opt.emb_size)

    img_encoder = conditional(
        models.cub_image.EncoderFT(latent_dim=opt.latent_dim if opt.deterministic else 2 * opt.latent_dim))

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

    print('{:.6f}'.format(groundtruth))

    if not opt.deterministic:
        acc_syn_cap, acc_syn_img = calc_synergy_coherence(paired_dataloader,
                                                          cap_encoder, cap_decoder,
                                                          img_encoder, img_decoder)
        print('{:.6f}'.format(acc_syn_cap))
        print('{:.6f}'.format(acc_syn_img))

    acc_c2i, acc_i2c = calc_cross_coherence(paired_dataloader, cap_encoder, cap_decoder, img_encoder, img_decoder)
    print('{:.6f}'.format(acc_c2i))
    print('{:.6f}'.format(acc_i2c))

    acc_joint = calc_joint_coherence(cap_decoder, img_decoder)
    print('{:.6f}'.format(acc_joint))


if __name__ == '__main__':
    main()
