# CS231N Final Project

We are using the dataset from the following Kaggle competition: [iMaterialist Challenge (Fashion) at FGVC5](https://www.kaggle.com/c/imaterialist-challenge-fashion-2018/).

The project proposal is available at [CS231N_Project_Proposal.pdf](./CS231N_Project_Proposal.pdf).

# Setup

#### Prerequisites

* **Python3**: we use Python version 3+ for this project.
* [Pipenv](https://github.com/pypa/pipenv): Python package manager and virtual environment. Can be installed with command `pip install pipenv`.

#### Initial Setup

At the first time, run the following commands:

```bash
git clone git@github.com:minfawang/cs231n-fashion.git  # Clones repo.
cd cs231n-fashion  # Changes your directory to the root of the repo.
# If you use a conda custom Python binary, then you may use the
# command in the comment below:
# pipenv --python /usr/local/bin/python3 install
pipenv --three install  # Create a virtual env using Python3.
```

#### Download data

First, download the `json` files from the Kaggle [data page](https://www.kaggle.com/c/imaterialist-challenge-fashion-2018/data).

Please download the files into **data/** directory and then unzip all of them. Then download the images using the script below:

```bash
# Change max_download parameters in the file.
python utils/downloader.py
```

#### Each run

Everytime you need to update the project or run the scripts:

```bash
pipenv shell  # Enter the virtual env.
# Make updates.
exit
```


#### Useful commands
```bash
# this will download the test_prediction to local
gcloud compute scp binbinx@cs231n-fashion-ssd:/home/shared/cs231n-fashion/submission/test_prediction.csv .
```

```bash
# after logging in, run the following command to monitor memory usage
sh /home/binbinx/memusg.sh
```
