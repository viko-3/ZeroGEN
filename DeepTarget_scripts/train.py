import argparse
import os
import sys
import torch
import rdkit

from DeepTarget.distribute_utils import init_distributed_mode, dist, cleanup
from DeepTarget.script_utils import add_train_args, read_smiles_csv, set_seed
from DeepTarget.models_storage import ModelsStorage
from DeepTarget.dataset import get_dataset

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

MODELS = ModelsStorage()


def get_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        title='Models trainer script', description='available models'
    )
    for model in MODELS.get_model_names():
        add_train_args(
            MODELS.get_model_train_parser(model)(
                subparsers.add_parser(model)
            )
        )
    return parser


def main(model, config):
    set_seed(config.seed)
    device = torch.device(config.device)
    ###
    init_distributed_mode(args=config)
    rank = config.rank
    config.lr *= config.world_size  # 学习率要根据并行GPU的数量进行倍增

    if config.config_save is not None:
        torch.save(config, config.config_save)

    # For CUDNN to work properly
    if device.type.startswith('cuda'):
        torch.cuda.set_device(device.index or 0)
    if config.train_load is None:
        train_data_smi, train_data_prot, train_data_mol_idx, train_data_prot_idx = get_dataset('train')
    else:
        train_data_smi, train_data_prot, train_data_mol_idx, train_data_prot_idx = read_smiles_csv(config.train_load)
    if config.val_load is None:
        val_data_smi, val_data_prot, val_data_mol_idx, val_data_prot_idx = get_dataset('test')
    else:
        val_data_smi, val_data_prot, val_data_mol_idx, val_data_prot_idx = read_smiles_csv(config.val_load)
    trainer = MODELS.get_model_trainer(model)(config)

    if config.vocab_load is not None:
        assert os.path.exists(config.vocab_load), \
            'vocab_load path does not exist!'
        vocab = torch.load(config.vocab_load)
    else:
        vocab = trainer.get_vocabulary(train_data_smi, train_data_prot)

    if config.vocab_save is not None:
        torch.save(vocab, config.vocab_save)

    train_data = [train_data_smi, train_data_prot, train_data_mol_idx, train_data_prot_idx]
    val_data = [val_data_smi, val_data_prot, val_data_mol_idx, val_data_prot_idx]

    model = MODELS.get_model_class(model)(vocab, config).to(device)
    # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
    if rank == 0:
        torch.save(model.state_dict(), config.model_save[:-3] + '_untrain.pt')
        dist.barrier()
        model.load_state_dict(torch.load(config.model_save[:-3] + '_untrain.pt', map_location=device))
    trainer.fit(model, train_data, val_data)
    if rank == 0:
        model = model.to('cpu')
        torch.save(model.state_dict(), config.model_save)
    cleanup()


if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_args()
    model = sys.argv[1]
    main(model, config)
