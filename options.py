import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--name', type=str, default='mnist_svhn', help='name of the experiment')
parser.add_argument('--outroot', type=str, default='/tmp', help='where to save the results')
parser.add_argument('--dataroot', type=str, default='/tmp/data', help='root directory of datasets')
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--checkpoint', type=str, help='pretrained checkpoint')
parser.add_argument('--checkpoint_dir', type=str, help='directory containing pretrained checkpoints')
parser.add_argument('--save_image', action='store_true', help='if save images to disk')

parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
parser.add_argument('--save_interval', type=int, default=2000, help='interval between samples')
parser.add_argument('--eval_interval', type=int, default=50000, help='interval between evaluation')
parser.add_argument('--no_eval', action='store_true', help='do not evaluate during training')

parser.add_argument('--max_iter', type=int, default=250000, help='maximum iterations of training')
parser.add_argument('--dis_iter', type=int, default=1)
parser.add_argument('--gen_iter', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate of Adam.')
parser.add_argument('--b1', type=float, default=0.5, help='beta1 (betas[0]) value of Adam')
parser.add_argument('--b2', type=float, default=0.999, help='beta2 (betas[1]) value of Adam')
parser.add_argument('--ema_start', type=int, default=50000)
parser.add_argument('--beta', type=float, default=0.9999)

parser.add_argument('--lambda_x_rec', type=float, default=0.0)
parser.add_argument('--lambda_c_rec', type=float, default=0.0)
parser.add_argument('--lambda_s_rec', type=float, default=0.0)
parser.add_argument('--lambda_unimodal', type=float, default=1.0)

parser.add_argument('--latent_dim', type=int, default=20, help='dimensionality of the whole latent space')
parser.add_argument('--style_dim', type=int, default=0, help='dimensionality of the style code')
parser.add_argument('--deterministic', action='store_true', help='use deterministic encoder')
parser.add_argument('--joint_posterior', action='store_true')

parser.add_argument('--content_only', action='store_true')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')

parser.add_argument('--data_multiplication', type=int, default=30)
parser.add_argument('--max_d', type=int, default=10000)
parser.add_argument('--use_all', action='store_true')

parser.add_argument('--n_extra', type=int, default=0)
parser.add_argument('--n_extra_x1', type=int, default=0)
parser.add_argument('--n_extra_x2', type=int, default=0)

parser.add_argument('--emb_size', type=int, default=128)
parser.add_argument('--joint_rec', action='store_true')
