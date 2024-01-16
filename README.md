## BENO: Boundary-embedded neural operators for elliptic PDEs





# Installation

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



# Dataset

The dataset files 10 4-Corners examples are under "data/". 

# Training

Below we provide example commands for training BENO. 

An example 4-Corners dataset training command is:

```code
python train.py --dataset_type=32x32 --epochs 1000
```


# Analysis

To analyze the results, use the following commands:

```code
python analysis.py 
```
