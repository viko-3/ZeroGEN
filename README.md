# ZeroGEN
Deep generative methods can generate new data that resemble a given distribution of data and have started to gain traction in ligand design. However, the ability of existing models to generate ligands for unseen target (referred to as zero-shot scenarios) present challenges. In this study, we introduce ZeroGEN, a novel zero-shot deep generative framework based on protein sequence. ZeroGEN incorporates contrastive learning to align the known protein-ligand feature, enhancing the comprehension of potential interactions between protein and ligands. Furthermore, ZeroGEN employs self-distillation to filter the initially generated data, remaining what the model deems as reliable new ligands and implementing data augmentation to help the model understand which ligands match unseen targets. The experimental results demonstrate that ZeroGEN is able to generate ligands for unseen targets with good affinity and drug-like properties. Additionally, the visualization of molecular docking and the attention matrix demonstrate that ZeroGEN has the ability to autonomously focus on key residues of proteins.

## Quickly start

### Main requirments
- Python 3.7
- rdkit 2020.09.1.0
- torch 1.11.0
- numpy
- pandas

### Data
Due to the size of the file, we will provide the available download path later, or you can contact us directly.

### Train
```python
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 9999 --use_env ZeroGEN_scripts/run.py --model ZeroGEN --checkpoint_dir save_checkpoints_file --train_path train_data_file.csv  --test_path test_data_file.csv
```

### Sample
```python
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 9999 --use_env ZeroGEN_scripts/sample.py --proteins_path proteins_path.csv --checkpoint_path checkpoints.pt --load_config_path checkpoints_config.pt --load_vocab_path checkpoints_vocab.pt --save_attn_matrix False
```
