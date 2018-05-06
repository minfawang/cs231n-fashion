# CS231N Final Project

We are using the dataset from the following Kaggle competition: [iMaterialist Challenge (Fashion) at FGVC5](https://www.kaggle.com/c/imaterialist-challenge-fashion-2018/).

The project proposal is available at [CS231N_Project_Proposal.pdf](./CS231N_Project_Proposal.pdf).

# Setup

Prerequisites:

* **Python3**: we use Python version 3+ for this project.
* [Pipenv](https://github.com/pypa/pipenv): Python package manager and virtual environment. Can be installed with command `pip install pipenv`.

At the first time, run the following commands:

```bash
git clone git@github.com:minfawang/cs231n-fashion.git  # Clones repo.
cd cs231n-fashion  # Changes your directory to the root of the repo.
# If you use a conda custom Python binary, then you may use the
# command in the comment below:
# pipenv --python /usr/local/bin/python3 install
pipenv --three install  # Create a virtual env using Python3.
```

Everytime you need to update the project or run the scripts:

```bash
pipenv shell  # Enter the virtual env.
# Make updates.
exit
```
