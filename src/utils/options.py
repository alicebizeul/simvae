import argparse

arg_attribs = {
    "b": ("beta_start",       ("04.2f", "<5.2f"), "\t",   False,  True),
    "e": ("epochs",     ("03d", "<4d"),     "\t",   False,  False),
    "E": ("eval_epochs",("03d", "<4d"),     "\t",   False,  False),
    "l": ("num_layer",  ("1s",  "1s"),      "\t",   False,  False),
    "N": ("dec_bn_off", ("1d", "d"),        "\t",   True,   False),
    "v": ("log_var_x_start",  ("04.2f", "<#5.2f"),"\t",   False,  True),
    "d": ("ds_name",    ("3s", "<10s"),     "\t",   False,  False),
    "c": ("num_class",  ("05d", "<6d"),     "\t",   False,  False),
    "a": ("n_augments", ("05d", "<6d"),     "\t",   False,  False),
    "S": ("sup_type",   ("1d", "d"),        "\t",   False,  False),
    "B": ("bs",         ("04d", "<4d"),     "\t\t", False,  False),
    "L": ("lr",         ("08.2e", "#8.2e"), "\t",   False,  True),
    "R": ("lr_eval",    ("08.2e", "#8.2e"), "\t",   False,  True),
    "s": ("seed",       ("05d", "<6d"),     "\t",   False,  False),
    "Z": ("zdim",      ("03d", "<4d"),     "\t",   False,  False),
    "F": ("eval_freq",  ("04d", "<4d"),     "\t",   False,  False),
    "o": ("p_y_prior",    ("3s", "<10s"),     "\t",   False,  False),
    "u": ("lr_schedule",    ("3s", "<10s"),     "\t",   False,  False),
}

parser = argparse.ArgumentParser()
parser.add_argument( "--no_train",          default=False,              action="store_true",    help="plot graphs from existing data, else train model", )
parser.add_argument( "-w", "--num_workers", default=4,     type=int,   action="store") 
parser.add_argument( "-b", "--beta_start",  default=0.15,   type=float, action="store",         help="beta of beta-VAE; weighting of KL term in loss", )
parser.add_argument( "-e", "--epochs",      default=400,     type=int,   action="store",         help="number of epochs to train the VAE for", )
parser.add_argument( "-l", "--num_layer",   default='2',    type=str,   action="store",         help="set num_layer (default 2, else 1) or (R)esnet", )
parser.add_argument( "-N", "--dec_bn_off",  default=False,              action="store_true",    help="turn off decoder batch norm layers" )
parser.add_argument( "-v", "--log_var_x_start",   default=-4.6,    type=float, action="store",         help="log variance of p(x|z) (default 0., ≥10 => learned)", )
parser.add_argument( "-d", "--ds_name",     default="mnist",type=str,       choices=["gaussian","mnist","celeba","cifar10","stl10","svhn","omniglot","fashionmnist","flowers","tinyimagenet","cub","cifar100","imagenet","wiki","utk"], help="dataset", )
parser.add_argument( "-c", "--num_class",   default=10,      type=int,   action="store",         help="set num_classes to generate (default 1)", )
parser.add_argument( "-B", "--bs",          default=128,    type=int,   action="store",         help="batch size for training" )
parser.add_argument( "-L", "--lr",          default=8e-5,   type=float, action="store",         help="learning rate" )
parser.add_argument( "-D", "--lr_decay",    default=1.0,    type=float, action="store",         help="learning rate decay forexponential scheduler" )
parser.add_argument( "-a", "--n_augments",  default=2,      type=int,   action="store",         help="number of augmentations for each sample", )
parser.add_argument( "-S", "--sup_type",    default=1,      type=int,   action="store",         help="0=GMMVAE, 1=SelfSup, 2=Supervised, 3=InfoNCE" )
parser.add_argument( "-s", "--seed",        default=1234,   type=int,   metavar="S",            help="random seed" )
parser.add_argument( "-p", "--path_output",   default="./ouputs",        help="path for storage of results", )
parser.add_argument( "-P", "--path_data",     default="./data",             help="path for fetching data", )
parser.add_argument( "-E", "--eval_epochs",   default=200,     type=int,   action="store",         help="num epochs for final train and validation of head")
parser.add_argument( "-R", "--lr_eval",       default=3e-4,   type=float,  action="store",         help="lr for the training of evaluation classifiers")
parser.add_argument( "-W", "--wt_decay",      default=0.,     type=float, action="store",         help="weight decay in optimizer")
parser.add_argument( "-G", "--trans_strength",default=None,   type=int,     help="add jitter and small crop or not")
parser.add_argument( "-H", "--checkpoint",    default=None,   type=str,   action="store",         help="path to checkpoint to load")
parser.add_argument( "-Z", "--zdim",          default=10,     type=int,   action="store",         help="dimension of latent space")
parser.add_argument( "-u", "--lr_schedule",   default="None", type=str, choices=["None","linear","cosine"], help="adding linear lr decreasy")
parser.add_argument( "-F", "--eval_freq",     default=20,     type=int,    help="dimension of latent space")
parser.add_argument( "-o", "--p_y_prior",     default="gaussian",type=str,       choices=["gaussian","uniform","MoG"], help="py prior type", )
parser.add_argument(       "--eval_duration", default=5,     type=int,    help="dimension of latent space")
parser.add_argument(       "--target_transform",     default="hair",type=str, help="dataset", )
parser.add_argument(       "--cuda",                 default=True,               action="store_false",   help="on cpu")

args = vars(parser.parse_args())

title_attribs, fn_attr_str = "", ""
for fn_tag, (k, formats, spacer, boolean, isfloat) in arg_attribs.items():
    v = args[k]
    if boolean:
        v = v * 1
    f0, f1 = ["%" + f for f in formats]
    fn_attr_str += f"{fn_tag}{v:{formats[0]}}_"
    title_attribs += f"{fn_tag}: {v:{formats[1]}}{spacer}"
fn_attr_str = fn_attr_str[:-1]
