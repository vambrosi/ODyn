# ODyn

## Installation

- Install a code editor with Python and TOML extensions (_e.g._ [VS Code](https://code.visualstudio.com/), its [python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python), and [Even Better TOML](https://marketplace.visualstudio.com/items?itemName=tamasfe.even-better-toml)).

- Open the terminal app and go to the folder where you want to create a clone of this repository. If you are on Mac and using the default GitHub folder, use this command in the terminal:

```bash
cd ~/Documents/GitHub
```

- Clone odyn (you can also use the GitHub app):

```bash
git clone https://github.com/vambrosi/ODyn.git
```

- Install `miniconda` by downloading the installer (for Windows) or running these commands on the Mac terminal (answer `yes` in the next two prompts)
```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

- Lastly, create a new conda environment and install `caiman` (from my fork) and `tomlkit`:

```
conda deactivate
git clone -b temp-fix --single-branch https://github.com/vambrosi/CaImAn.git
cd CaImAn/
mamba env create -f environment.yml -n caiman_va
mamba activate caiman_va
pip install -e .
pip install tomlkit
```

## Usage

If you are using VS Code, open the `odyn` folder, go to `View -> Command Palette`, type `Python: Select Interpreter`, and pick `Python 3.12.0 (caiman_va)` from the list. Then, make a copy of `DONT_EDIT_processing.ipynb` and follow the instructions there. You will possibly need to select `caiman_va` as your kernel and install the Jupyter extension (you will get prompts after you try to run a notebook cell).