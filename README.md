# APM-DETR
Adaptive Perceptual Multi-scale DETR for Road Defect Detection in Complex Scenes


# Environment

In this experiment, we employed an NVIDIA GeForce RTX 3090 GPU (equipped with 24124 MiB video memory) for computational acceleration, and the CUDA 12.1 toolkit was utilized to facilitate parallel computing in deep learning tasks. The software stack was configured as follows: Python 3.10.16 as the programming language, and PyTorch 2.2.2 (compatible with CUDA 12.1, denoted as torch-2.2.2+cu121) as the deep learning framework.

Install conda and create a conda environment:

conda create -n APM-DETR
conda activate APM-DETR
pip install -r requirements.txt

# Training


$ python train.py

# Test



$ python detect.py

# Val



$ python val.py

# Datsets


The RDD2022 (China-D), and UAV-PDD2023 datasets can be downloaded with:

RDD2022:https://github.com/sekilab/RoadDamageDetector

UAV-PDD2023:https://zenodo.org/records/8429208
