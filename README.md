# ZeroGEN
Deep generative methods can generate new data that resemble a given distribution of data and have started to gain traction in ligand design. However, the ability of existing models to generate ligands for unseen target (referred to as zero-shot scenarios) present challenges. In this study, we introduce ZeroGEN, a novel zero-shot deep generative framework based on protein sequence. ZeroGEN incorporates contrastive learning to align the known protein-ligand feature, enhancing the comprehension of potential interactions between protein and ligands. Furthermore, ZeroGEN employs self-distillation to filter the initially generated data, remaining what the model deems as reliable new ligands and implementing data augmentation to help the model understand which ligands match unseen targets. The experimental results demonstrate that ZeroGEN is able to generate ligands for unseen targets with good affinity and drug-like properties. Additionally, the visualization of molecular docking and the attention matrix demonstrate that ZeroGEN has the ability to autonomously focus on key residues of proteins.

## Quickly start

### Main requirments
- Python 3.7
- rdkit 2020.09.1.0
- torch 1.11.0
- numpy
- pandas

### Train
```python
python test.py
```

### Sample
```python
python test.py
```