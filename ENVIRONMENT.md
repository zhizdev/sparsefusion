# Environment Setup
We describe steps (in Linux command line) to setup the environment for SparseFusion. 

## Conda Environment
We install and setup a conda environment. 

### (optional) Install Conda
Required if conda not installed. 
```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
export PATH="/home/username/miniconda/bin:$PATH"
conda init
source ~/.bashrc
```

### Create New Environment
```bash
conda create -n sparsefusion python=3.8
conda activate sparsefusion
```

## Install Dependencies
We install the necessary dependencies. 

### GCC and Cuda
Make sure to do this first!

We also assume that nvidia drivers and `cuda=11.3.x` is installed.
```bash
conda install -c conda-forge cxx-compiler=1.3.0
conda install -c conda-forge cudatoolkit-dev
conda install -c conda-forge ninja
```

### Python Libraries
```bash
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c pytorch3d pytorch3d
```

### Support Stable Diffusion
```bash
pip install transformers==4.19.2 pytorch-lightning==1.4.2 torchmetrics==0.6.0
pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
```

### (optional) Install CO3D
Required if using full CO3D dataset.
```bash
git clone https://github.com/facebookresearch/co3d
cd co3d
pip install -r requirements.txt
pip install -e .
```

### Install Other SparseFusion Requirements
```bash
cd sparsefusion
pip install -r requirements.txt
```

## Building Extensions 
We require a few extensions from [torch-ngp](https://github.com/ashawkey/torch-ngp). We detail how to install them below. See more details on the [torch-ngp](https://github.com/ashawkey/torch-ngp) Github. 

### Build gridencoder
```bash
pip install ./external/gridencoder
```

### Build raymarcher
```bash
pip install ./raymarching
```