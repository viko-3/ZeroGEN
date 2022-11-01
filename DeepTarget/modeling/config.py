import argparse


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    # Model
    model_arg = parser.add_argument_group('Model')
    model_arg.add_argument("--alpha", type=int, default=0.2,
                           help="momen alpha value")
    model_arg.add_argument("--ninp", type=int, default=128,
                           help="size of word embeddings")
    model_arg.add_argument("--prot_dim", type=int, default=1024,
                           help="size of prot model previous embeddings")
    model_arg.add_argument("--attn_dim", type=int, default=128,
                           help="size of attention dim")
    model_arg.add_argument("--num_layers", type=int, default=4,
                           help="Number of Transformer layers")
    parser.add_argument('--head', type=int, default=4,
                        help='the number of heads in the encoder/decoder of the transformer model')
    model_arg.add_argument("--hidden", type=int, default=256,
                           help="number of hidden units per layer")
    model_arg.add_argument("--dropout", type=float, default=0.2,
                           help="dropout between Transformer layers except for last")

    # Train
    train_arg = parser.add_argument_group('Training')
    train_arg.add_argument('--train_epochs', type=int, default=100,
                           help='Number of epochs for model training')
    train_arg.add_argument('--n_batch', type=int, default=16,
                           help='Size of batch')
    train_arg.add_argument('--lr', type=float, default=1e-4,
                           help='Learning rate')
    train_arg.add_argument('--step_size', type=int, default=10,
                           help='Period of learning rate decay')
    train_arg.add_argument('--gamma', type=float, default=0.5,
                           help='Multiplicative factor of learning rate decay')
    train_arg.add_argument('--n_jobs', type=int, default=8,
                           help='Number of threads')
    train_arg.add_argument('--n_workers', type=int, default=1,
                           help='Number of workers for DataLoaders')

    train_arg.add_argument('--multi_gpu', default=False, type=bool,
                           help='Parallel or not')

    train_arg.add_argument('--load_pretrain', default=True, type=bool,
                           help='load_pretrain model or not')

    train_arg.add_argument('--MLM_model', default=False, type=bool,
                           help='train MLM model or not')
    train_arg.add_argument('--CL_model', default=False, type=bool,
                           help='train contrastive learning model or not')
    train_arg.add_argument('--Matching_model', default=False, type=bool,
                           help='train matching model or not')
    train_arg.add_argument('--Language_model', default=True, type=bool,
                           help='train language model or not')

    train_arg.add_argument('--pretrain_Language_model', default=False, type=bool,
                           help='pretrain language model or not')
    train_arg.add_argument('--lm_pretrain_load', default='data/LM_pretrain.csv', type=str,
                           help='pretrain language model data')
    train_arg.add_argument('--lm_pretrain_val_load', default='data/zinc_test.csv', type=str,
                           help='pretrain language model val data')

    ###
    # 不要改该参数，系统会自动分配
    # train_arg.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    # 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
    train_arg.add_argument('--world_size', default=4, type=int,
                           help='number of distributed processes')
    train_arg.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser


def get_config():
    parser = get_parser()
    return parser.parse_known_args()[0]
