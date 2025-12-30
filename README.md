# ODyn

## Installation

Open the terminal and go to the folder where you want to create a clone of this repository.

Clone odyn (you can also use the GitHub app):
```
git clone https://github.com/vambrosi/ODyn.git
```

Create a new conda environment and install `caiman` (from my fork) and `tomlkit`:
```
git clone -b temp-fix --single-branch https://github.com/vambrosi/CaImAn.git
cd CaImAn/
mamba env create -f environment.yml -n caiman_va
conda deactivate
mamba activate caiman_va
pip install -e .
pip install tomlkit
```