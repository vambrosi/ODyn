# ODyn

## Installation

- Install a code editor with a Python extensions (_e.g._ [VS Code](https://code.visualstudio.com/) and its [python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python)).

- Install `git` if it's not installed yet. This can be done by downloading the [GitHub Desktop](https://desktop.github.com/download/) app, or intalling `git` directly from the [official website](https://git-scm.com/).

- Install `miniforge` by downloading the [installer](https://github.com/conda-forge/miniforge?tab=readme-ov-file#windows) (for Windows) or running these commands on the Mac terminal (answer `yes` in the next two prompts)
```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

- Create a new conda environment and install `caiman`. You can do that by opening a terminal (`Miniforge Prompt` in Windows), and copy-pasting the following:

```bash
conda deactivate
mamba create -n caiman caiman
```

- Clone the `odyn` repository. If you have VS Code, you can create a new window, go to _Source Control_ in your left toolbar, click on _Clone Repository_, choose _Clone from GitHub_, and write `vambrosi/ODyn` (you can pick any folder you would like).

- Lastly, install `odyn` using `pip`. To do that, open a terminal that has `conda` (`Miniforge Prompt` in Windows, or the default terminal in MacOS) and run:

```bash
conda activate caiman
pip install -e ODYN_PATH
```

where `ODYN_PATH` is the folder where you installed `odyn`.

## Usage

Create a notebook file (extension `.ipynb`) in VS Code, and run `from odyn import Database` in a cell. You will get more instructions after that. You will also need to select `caiman` as your _Python Environment_ and install the Jupyter extension (you will get prompts after you try to run a notebook cell).
