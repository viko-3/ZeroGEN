import argparse
import random
import re
import numpy as np
import pandas as pd
import torch


def add_common_arg(parser):
    def torch_device(arg):
        if re.match('^(cuda(:[0-9]+)?|cpu)$', arg) is None:
            raise argparse.ArgumentTypeError(
                'Wrong device format: {}'.format(arg)
            )

        if arg != 'cpu':
            splited_device = arg.split(':')

            if (not torch.cuda.is_available()) or \
                    (len(splited_device) > 1 and
                     int(splited_device[1]) > torch.cuda.device_count()):
                raise argparse.ArgumentTypeError(
                    'Wrong device: {} is not available'.format(arg)
                )

        return arg

    # Base
    parser.add_argument('--device',
                        type=torch_device, default='cuda',
                        help='Device to run: "cpu" or "cuda:<device number>"')
    parser.add_argument('--seed',
                        type=int, default=0,
                        help='Seed')

    return parser


def add_train_args(parser):
    # Common
    common_arg = parser.add_argument_group('Common')
    add_common_arg(common_arg)
    common_arg.add_argument('--train_load',
                            type=str,
                            help='Input data in csv format to train')
    common_arg.add_argument('--val_load', type=str,
                            help="Input data in csv format to validation")
    common_arg.add_argument('--model_save',
                            type=str, required=True, default='model.pt',
                            help='Where to save the model')
    common_arg.add_argument('--save_frequency',
                            type=int, default=5,
                            help='How often to save the model')
    common_arg.add_argument('--log_file',
                            type=str, required=False,
                            help='Where to save the log')
    common_arg.add_argument('--config_save',
                            type=str, required=True,
                            help='Where to save the config')
    common_arg.add_argument('--vocab_save',
                            type=str,
                            help='Where to save the vocab')
    common_arg.add_argument('--vocab_load',
                            type=str,
                            help='Where to load the vocab; '
                                 'otherwise it will be evaluated')
    common_arg.add_argument("--max_len",
                            type=int, default=300,
                            help="Max of length of SMILES")
    return parser


def add_sample_args(parser):
    # Common
    common_arg = parser.add_argument_group('Common')
    add_common_arg(common_arg)
    common_arg.add_argument('--model_load',
                            type=str, required=True,
                            help='Where to load the model')
    common_arg.add_argument('--config_load',
                            type=str, required=True,
                            help='Where to load the config')
    common_arg.add_argument('--vocab_load',
                            type=str, required=True,
                            help='Where to load the vocab')
    common_arg.add_argument('--n_samples',
                            type=int, required=True,
                            help='Number of samples to sample')
    common_arg.add_argument('--gen_save',
                            type=str, required=True,
                            help='Where to save the gen molecules')
    common_arg.add_argument("--n_batch",
                            type=int, default=100,
                            help="Size of batch")

    return parser


def read_smiles_csv(path):
    df = pd.read_csv(path)
    # df = df.sample(n=100000, random_state=123, replace=True)
    mol = df['SMILES'].astype(str).tolist()
    prot = df['protein_seq'].astype(str).tolist()
    mol_idx = df['mol_idx'].astype(int).tolist()
    prot_idx = df['cluster_id'].astype(int).tolist()
    affinity_score = df['pKd_pKi_pIC50'].astype(float).tolist()

    max_value, min_value = max(affinity_score), min(affinity_score)
    affinity_score = df['pKd_pKi_pIC50'].apply(clamp_affinity_score, args=(max_value, min_value))

    return mol, prot, mol_idx, prot_idx, affinity_score


def read_pretain_lm_csv(path):
    df = pd.read_csv(path)
    mol = df['SMILES'].astype(str).tolist()[:10000000]
    return mol


def clamp_affinity_score(x, max_value, min_value):
    clamp_value = (x - min_value) / (max_value - min_value)
    return clamp_value


def read_proteins_csv(path):
    prot = pd.read_csv(path,
                       usecols=['protein_seq'],
                       squeeze=True).astype(str).tolist()[:100]
    prot_idx = pd.read_csv(path,
                           usecols=['prot_idx'],
                           squeeze=True).astype(int).tolist()[:100]
    return [prot, prot_idx]


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
