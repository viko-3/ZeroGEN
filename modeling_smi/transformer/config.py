import argparse


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    # Model
    model_arg = parser.add_argument_group('Model')
    model_arg.add_argument("--ninp", type=int, default=128,
                           help="size of word embeddings")
    model_arg.add_argument("--num_layers", type=int, default=4,
                           help="Number of Transformer layers")
    parser.add_argument('--head', type=int, default=8,
                        help='the number of heads in the encoder/decoder of the transformer model')
    model_arg.add_argument("--hidden", type=int, default=256,
                           help="number of hidden units per layer")
    model_arg.add_argument("--dropout", type=float, default=0.2,
                           help="dropout between Transformer layers except for last")

    # Train
    train_arg = parser.add_argument_group('Training')
    train_arg.add_argument('--train_epochs', type=int, default=100,
                           help='Number of epochs for model training')
    train_arg.add_argument('--n_batch', type=int, default=256,
                           help='Size of batch')
    train_arg.add_argument('--lr', type=float, default=1e-3,
                           help='Learning rate')
    train_arg.add_argument('--step_size', type=int, default=10,
                           help='Period of learning rate decay')
    train_arg.add_argument('--gamma', type=float, default=0.5, 
                           help='Multiplicative factor of learning rate decay')
    train_arg.add_argument('--n_jobs', type=int, default=8,
                           help='Number of threads')
    train_arg.add_argument('--n_workers', type=int, default=1,
                           help='Number of workers for DataLoaders')

    return parser


def get_config():
    parser = get_parser()
    return parser.parse_known_args()[0]
