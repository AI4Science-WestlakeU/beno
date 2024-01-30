# BENO: Boundary-embedded Neural Operators for Elliptic PDEs (ICLR 2024)

[Paper](https://openreview.net/forum?id=ZZTkLDRmkg) | [arXiv](https://arxiv.org/abs/2401.09323) | [Tweet](https://twitter.com/tailin_wu/status/1747259448635367756)

[Haixin Wang*](https://willdreamer.github.io/), [Jiaxin Li*](https://github.com/Jiaxinlia/Jiaxin.github.io), [Anubhav Dwivedi](https://dwivedi-anubhav.github.io/website/), [Kentaro Hara](https://aa.stanford.edu/people/ken-hara), [Tailin Wu](https://tailin.org/)

We introduce a boundary-embedded neural operator that incorporates **complex boundary shape** and **inhomogeneous boundary values** shown below into the solving of Elliptic PDEs:

<a href="url"><img src="https://github.com/AI4Science-WestlakeU/beno/blob/main/assets/motivation.png" align="center" width="700" ></a>

The whole architecture of our BENO:

<a href="url"><img src="https://github.com/AI4Science-WestlakeU/beno/blob/main/assets/arch.png" align="center" width="700" ></a>

## Installation

1. First clone the directory.

2. Install dependencies.

First, create a new environment using [conda](https://docs.conda.io/en/latest/miniconda.html) (with python >= 3.7). Then install pytorch, torch-geometric and other dependencies as follows 

Install pytorch (replace "cu113" with appropriate cuda version. For example, cuda11.1 will use "cu111"):
```code
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

Install torch-geometric. Run the following command:
```code
pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.10.2+cu113.html
pip install torch-sparse==0.6.12 -f https://data.pyg.org/whl/torch-1.10.2+cu113.html
pip install torch-geometric==1.7.2
pip install torch-cluster==1.5.9 -f https://data.pyg.org/whl/torch-1.10.2+cu113.html
pip install loguru
```

## Dataset

The sample dataset files 10 4-Corners examples are under "data/". And the full dataset files can be downloaded via [this link](https://drive.google.com/file/d/11PbUrzJ-b18VhFGY_uICSciCkeGrsaTZ/view). To run experiments on specific boundary types, download the files in the link into the "data/" folder in the local repo. **BC_Nxx_xc_all.npy**/**RHS_Nxx_xc_all.npy**/**SOL_Nxx_xc_all.npy** represents **boundary information**/**forcing term**/**solution** of the specific resolution and shapes.


## Training

Below we provide example commands for training BENO. 

An example 4-Corners dataset training command is:

```code
python train.py --dataset_type=32x32 --epochs 1000
```


## Analysis

To analyze the results, use the following commands:

```code
python analysis.py 
```

## Related Projects:

* [CinDM](https://github.com/AI4Science-WestlakeU/cindm) (ICLR 2024 spotlight): We introduce a method that uses compositional generative models to design boundaries and initial states significantly more complex than the ones seen in training for physical simulations.

* [LAMP](https://github.com/snap-stanford/lamp) (ICLR 2023 spotlight): first fully DL-based surrogate model that jointly optimizes spatial resolutions to reduce computational cost and learns the evolution model, learned via reinforcement learning.


## Citation
If you find our work and/or our code useful, please cite us via:

```bibtex
@inproceedings{wang2024beno,
title={{BENO}: Boundary-embedded Neural Operators for Elliptic {PDE}s},
author={Wang, Haixin and Jiaxin, LI and Dwivedi, Anubhav and Hara, Kentaro and Wu, Tailin},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=ZZTkLDRmkg}
}
```

